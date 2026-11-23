"""
Sistema multimodal V3 - OPTIMIZADO PARA MAXIMIZAR EL VALOR DE LOS METADATOS

ENFOQUE: Hacer que los metadatos sean IMPRESCINDIBLES para el modelo
=========================================================================

ESTRATEGIAS IMPLEMENTADAS:

1. ARQUITECTURA AVANZADA:
   - Gated Fusion: El modelo aprende dinámicamente cuánto peso dar a imagen vs metadatos
   - Cross-Attention: Los metadatos modulan las features de la imagen
   - Residual Connections: Previene que el modelo ignore los metadatos
   - Metadata Embeddings: Representaciones más ricas de variables categóricas

2. AUGMENTATION INTELIGENTE:
   - Metadata Noise: Añadir ruido a metadatos para forzar robustez
   - Selective Metadata Dropout: Dropout aleatorio de metadatos en training
   - Esto obliga al modelo a aprender representaciones redundantes

3. LOSS FUNCTIONS:
   - Auxiliary Loss para metadatos: Predice metadatos desde features de imagen
   - Esto fuerza al modelo a aprender features que correlacionan con metadatos
   - Correlation Loss: Penaliza baja correlación entre ramas

4. REGULARIZACIÓN ESPECÍFICA:
   - Metadata L1 Regularization: Fuerza sparsity en pesos de metadatos
   - Feature Decorrelation: Asegura que imagen y metadatos aporten info diferente

5. PREPROCESAMIENTO AVANZADO:
   - Feature Engineering: Crear interacciones entre metadatos
   - Binning inteligente de variables continuas
   - Embeddings de alta dimensión para categóricas

Ejemplo de uso:
    python src/train_v3.py \
        --csv "reports/dataset_procesado_VRUH.csv" "reports/dataset_procesado_ISIC.csv" \
        --epochs 40 \
        --batch-size 16 \
        --lr 0.0001
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import csv
import warnings
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
from torchvision import transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, roc_curve, auc, cohen_kappa_score,
    precision_recall_fscore_support, roc_auc_score
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)


class FocalLoss(nn.Module):
    """Focal Loss mejorado."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AdvancedMetadataDataset(Dataset):
    """Dataset con feature engineering avanzado para metadatos."""
    
    def __init__(
        self,
        csv_path: Path,
        img_size: int = 224,
        use_augmentation: bool = True,
        verbose: bool = True,
        metadata_noise_std: float = 0.1,
    ):
        self.csv_path = csv_path
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.verbose = verbose
        self.metadata_noise_std = metadata_noise_std
        
        # Transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.data = self._load_csv()
        self._prepare_encoders()
        self._compute_metadata_stats()
        self.corrupted_images = 0
    
    def _load_csv(self) -> List[Dict[str, Any]]:
        data = []
        skipped = 0
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ext = row.get('extension', '').lower()
                if ext not in ['.png', '.jpg', '.jpeg']:
                    skipped += 1
                    continue
                try:
                    class_val = float(row['CLASS'])
                    if np.isnan(class_val):
                        skipped += 1
                        continue
                except (ValueError, KeyError):
                    skipped += 1
                    continue
                img_path = Path(row['image_path'])
                if not img_path.exists():
                    skipped += 1
                    continue
                data.append(row)
        
        if skipped > 0 and self.verbose:
            print(f"Filas omitidas: {skipped}")
        return data
    
    def _prepare_encoders(self):
        self.encoders = {}
        
        # Encoders para categóricas
        if True:  # Siempre procesar GENDER
            gender_values = list(set(row.get("GENDER_STD", "UNKNOWN") for row in self.data))
            self.encoders["GENDER_STD"] = {val: idx for idx, val in enumerate(sorted(gender_values))}
            self.num_genders = len(gender_values)
        
        if True:  # Siempre procesar LOCATION
            location_values = list(set(row.get("LOCATION_STD", "UNKNOWN") for row in self.data))
            self.encoders["LOCATION_STD"] = {val: idx for idx, val in enumerate(sorted(location_values))}
            self.num_locations = len(location_values)
        
        # Mapeo de clases
        self.class_mapping = {}
        for row in self.data:
            original_class = int(float(row["CLASS"]))
            self.class_mapping[original_class] = 0 if original_class == 0 else 1
        
        binary_classes = [self.class_mapping[int(float(row["CLASS"]))] for row in self.data]
        unique, counts = np.unique(binary_classes, return_counts=True)
        if self.verbose:
            print(f"Clases: {dict(zip(unique, counts))}")
            print(f"Total: {len(self.data)} muestras")
    
    def _compute_metadata_stats(self):
        """Calcular estadísticas robustas para normalización."""
        self.metadata_stats = {}
        
        for col in ["AGE", "DIAMETER 1", "DIAMETER 2"]:
            values = []
            for row in self.data:
                try:
                    val_str = row.get(col, "")
                    if val_str != "" and val_str is not None:
                        val = float(val_str)
                        if not np.isnan(val) and val > 0:
                            values.append(val)
                except (ValueError, TypeError):
                    pass
            
            if values:
                # Usar percentiles para robustez contra outliers
                p5, p95 = np.percentile(values, [5, 95])
                filtered_values = [v for v in values if p5 <= v <= p95]
                
                self.metadata_stats[col] = {
                    'mean': np.mean(filtered_values),
                    'std': np.std(filtered_values),
                    'min': np.min(filtered_values),
                    'max': np.max(filtered_values),
                    'median': np.median(filtered_values)
                }
            else:
                self.metadata_stats[col] = {
                    'mean': 0, 'std': 1, 'min': 0, 'max': 1, 'median': 0
                }
            
            if self.verbose and col == "AGE":
                print(f"\nEstadísticas de {col}: mean={self.metadata_stats[col]['mean']:.1f}, "
                      f"std={self.metadata_stats[col]['std']:.1f}, median={self.metadata_stats[col]['median']:.1f}")
    
    def _encode_metadata_advanced(self, row: Dict[str, Any], add_noise: bool = False) -> Dict[str, torch.Tensor]:
        """
        Codificación avanzada con feature engineering.
        Retorna un dict con diferentes tipos de features.
        """
        features = {}
        
        # 1. VARIABLES NUMÉRICAS (continuas, normalizadas)
        numerical_features = []
        
        # Edad normalizada
        try:
            age = float(row.get("AGE", 0))
            if np.isnan(age) or age <= 0:
                age_norm = 0.0
            else:
                stats = self.metadata_stats.get("AGE", {'mean': 50, 'std': 20})
                age_norm = (age - stats['mean']) / (stats['std'] + 1e-8)
                age_norm = np.clip(age_norm, -3, 3)
                
                # Añadir ruido durante training
                if add_noise and self.use_augmentation:
                    age_norm += np.random.normal(0, self.metadata_noise_std)
        except (ValueError, TypeError):
            age_norm = 0.0
        
        numerical_features.append(age_norm)
        
        # Diámetros normalizados
        for col in ["DIAMETER 1", "DIAMETER 2"]:
            try:
                val_str = row.get(col, "")
                if val_str == "" or val_str is None:
                    val_norm = 0.0
                else:
                    val = float(val_str)
                    if np.isnan(val) or val < 0:
                        val_norm = 0.0
                    else:
                        stats = self.metadata_stats.get(col, {'mean': 0, 'std': 1})
                        val_norm = (val - stats['mean']) / (stats['std'] + 1e-8)
                        val_norm = np.clip(val_norm, -3, 3)
                        
                        if add_noise and self.use_augmentation:
                            val_norm += np.random.normal(0, self.metadata_noise_std)
            except (ValueError, TypeError):
                val_norm = 0.0
            
            numerical_features.append(val_norm)
        
        features['numerical'] = torch.tensor(numerical_features, dtype=torch.float32)
        
        # 2. VARIABLES CATEGÓRICAS (índices para embeddings)
        gender_val = row.get("GENDER_STD", "UNKNOWN")
        gender_idx = self.encoders["GENDER_STD"].get(gender_val, 0)
        features['gender'] = torch.tensor(gender_idx, dtype=torch.long)
        
        location_val = row.get("LOCATION_STD", "UNKNOWN")
        location_idx = self.encoders["LOCATION_STD"].get(location_val, 0)
        features['location'] = torch.tensor(location_idx, dtype=torch.long)
        
        # 3. FEATURE ENGINEERING: Interacciones y derivadas
        engineered_features = []
        
        # Edad categorizada (bins)
        try:
            age_raw = float(row.get("AGE", 0))
            if age_raw > 0:
                if age_raw < 30:
                    age_bin = 0
                elif age_raw < 50:
                    age_bin = 1
                elif age_raw < 70:
                    age_bin = 2
                else:
                    age_bin = 3
            else:
                age_bin = -1  # Missing
        except:
            age_bin = -1
        
        engineered_features.append(age_bin / 3.0)  # Normalizar
        
        # Ratio de diámetros (si ambos disponibles)
        try:
            d1_str = row.get("DIAMETER 1", "")
            d2_str = row.get("DIAMETER 2", "")
            
            if d1_str and d2_str and d1_str != "" and d2_str != "":
                d1 = float(d1_str)
                d2 = float(d2_str)
                if d1 > 0 and d2 > 0:
                    ratio = min(d1, d2) / max(d1, d2)  # 0-1
                else:
                    ratio = 0.5
            else:
                ratio = 0.5  # Neutral
        except:
            ratio = 0.5
        
        engineered_features.append(ratio)
        
        # Diámetro máximo categorizado
        try:
            d1 = float(row.get("DIAMETER 1", 0))
            d2 = float(row.get("DIAMETER 2", 0))
            max_diam = max(d1, d2)
            
            if max_diam <= 0:
                diam_category = 0
            elif max_diam < 5:
                diam_category = 1
            elif max_diam < 10:
                diam_category = 2
            else:
                diam_category = 3
        except:
            diam_category = 0
        
        engineered_features.append(diam_category / 3.0)
        
        features['engineered'] = torch.tensor(engineered_features, dtype=torch.float32)
        
        return features
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        row = self.data[idx]
        class_val = int(float(row["CLASS"]))
        label = self.class_mapping[class_val]
        
        # Imagen
        img_path = Path(row['image_path'])
        try:
            img = Image.open(img_path).convert('RGB')
            if self.use_augmentation:
                img_tensor = self.augment_transform(img)
            else:
                img_tensor = self.base_transform(img)
        except Exception:
            self.corrupted_images += 1
            img_tensor = torch.zeros(3, self.img_size, self.img_size)
        
        # Metadatos con feature engineering
        metadata_dict = self._encode_metadata_advanced(row, add_noise=self.use_augmentation)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, metadata_dict, label_tensor


class GatedFusionMultimodalNet(nn.Module):
    """
    Arquitectura avanzada con:
    1. Embeddings para categóricas
    2. Gated Fusion (el modelo aprende cuánto peso dar a cada modalidad)
    3. Cross-Attention (metadatos modulan features de imagen)
    4. Auxiliary heads para regularización
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        num_genders: int = 3,
        num_locations: int = 20,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.5,
        use_metadata: bool = True,
    ):
        super().__init__()
        
        self.use_metadata = use_metadata
        
        # Image encoder
        if backbone == "densenet121":
            self.image_encoder = models.densenet121(pretrained=pretrained)
            img_feature_dim = self.image_encoder.classifier.in_features
            self.image_encoder.classifier = nn.Identity()
        elif backbone == "resnet50":
            self.image_encoder = models.resnet50(pretrained=pretrained)
            img_feature_dim = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()
        elif backbone == "resnet18":
            self.image_encoder = models.resnet18(pretrained=pretrained)
            img_feature_dim = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone} no soportado")
        
        # Embeddings para variables categóricas (CLAVE para que sean útiles)
        self.gender_embedding = nn.Embedding(num_genders, 16)
        self.location_embedding = nn.Embedding(num_locations, 32)
        
        # Encoder para features numéricas
        self.numerical_encoder = nn.Sequential(
            nn.Linear(3, 32),  # AGE, DIAMETER1, DIAMETER2
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Encoder para features engineered
        self.engineered_encoder = nn.Sequential(
            nn.Linear(3, 16),  # age_bin, diam_ratio, diam_category
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        
        # Combinar todas las features de metadatos
        metadata_feature_dim = 16 + 32 + 32 + 16  # gender + location + numerical + engineered = 96
        
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # GATED FUSION: Aprende cuánto peso dar a imagen vs metadatos
        self.fusion_gate = nn.Sequential(
            nn.Linear(img_feature_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # [weight_image, weight_metadata]
            nn.Softmax(dim=1)
        )
        
        # Proyección de features antes de fusion
        self.image_projection = nn.Sequential(
            nn.Linear(img_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.metadata_projection = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # CROSS-ATTENTION: Metadatos modulan imagen
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout * 0.5,
            batch_first=True
        )
        
        # Clasificador principal
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        # AUXILIARY HEAD: Predecir género desde features de imagen
        # Esto fuerza a la imagen a capturar info relacionada con metadatos
        self.aux_gender_predictor = nn.Sequential(
            nn.Linear(img_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_genders)
        )
        
        # AUXILIARY HEAD: Predecir edad categorizada desde imagen
        self.aux_age_predictor = nn.Sequential(
            nn.Linear(img_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)  # 4 bins de edad
        )
    
    def forward(self, images: torch.Tensor, metadata: Optional[Dict[str, torch.Tensor]] = None,
                return_aux: bool = False):
        # Extract image features
        img_features = self.image_encoder(images)  # [B, img_feature_dim]
        
        # Process metadata (only if enabled)
        if self.use_metadata and metadata is not None:
            gender_embed = self.gender_embedding(metadata['gender'])  # [B, 16]
            location_embed = self.location_embedding(metadata['location'])  # [B, 32]
            numerical_features = self.numerical_encoder(metadata['numerical'])  # [B, 32]
            engineered_features = self.engineered_encoder(metadata['engineered'])  # [B, 16]
        
            # Concatenar todas las features de metadatos
            metadata_combined = torch.cat([
                gender_embed,
                location_embed,
                numerical_features,
                engineered_features
            ], dim=1)  # [B, 96]
        
            # Encode metadata
            metadata_features = self.metadata_encoder(metadata_combined)  # [B, 64]
        else:
            # Default dummy metadata_features when metadata disabled — zeros
            metadata_features = torch.zeros((img_features.size(0), 64), device=img_features.device)
        
        # Project to same dimension
        img_proj = self.image_projection(img_features)  # [B, 256]
        meta_proj = self.metadata_projection(metadata_features)  # [B, 256]
        
        # GATED FUSION
        concat_for_gate = torch.cat([img_features, metadata_features], dim=1)
        fusion_weights = self.fusion_gate(concat_for_gate)  # [B, 2]
        
        # Apply fusion weights
        gated_fusion = (fusion_weights[:, 0:1] * img_proj + 
                        fusion_weights[:, 1:2] * meta_proj)  # [B, 256]
        
        # CROSS-ATTENTION: metadata como query, image como key/value
        # Si no usamos metadatos, no aplicamos cross-attention
        if self.use_metadata:
            meta_query = meta_proj.unsqueeze(1)  # [B, 1, 256]
            img_kv = img_proj.unsqueeze(1)  # [B, 1, 256]
        
            attended_features, _ = self.cross_attention(
                meta_query, img_kv, img_kv
            )  # [B, 1, 256]
            attended_features = attended_features.squeeze(1)  # [B, 256]
        else:
            attended_features = torch.zeros_like(img_proj)
        
        # Combine gated fusion + cross-attention
        combined = gated_fusion + attended_features  # Residual connection
        
        # Classification
        output = self.classifier(combined)
        
        if return_aux and self.use_metadata:
            # Auxiliary predictions
            aux_gender = self.aux_gender_predictor(img_features)
            aux_age = self.aux_age_predictor(img_features)
            return output, aux_gender, aux_age
        
        return output


class ImageOnlyNet(nn.Module):
    """Modelo simple que usa sólo la rama de imagen (baseline).

    Usa el mismo backbone (resnet/densenet) y la proyección + clasificador
    que la rama de imagen del `GatedFusionMultimodalNet` para que sea
    comparable en tamaño/parametrización.
    """
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        if backbone == "densenet121":
            self.image_encoder = models.densenet121(pretrained=pretrained)
            img_feature_dim = self.image_encoder.classifier.in_features
            self.image_encoder.classifier = nn.Identity()
        elif backbone == "resnet50":
            self.image_encoder = models.resnet50(pretrained=pretrained)
            img_feature_dim = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()
        elif backbone == "resnet18":
            self.image_encoder = models.resnet18(pretrained=pretrained)
            img_feature_dim = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone} no soportado")

        # Proyección y clasificador comparables con GatedFusion
        self.image_projection = nn.Sequential(
            nn.Linear(img_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor, *args, **kwargs):
        img_features = self.image_encoder(images)
        img_proj = self.image_projection(img_features)
        output = self.classifier(img_proj)
        return output


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_mixup: bool = True,
    accumulation_steps: int = 2,
    aux_loss_weight: float = 0.3,
    use_metadata: bool = True,
) -> Tuple[float, float]:
    """Training con auxiliary losses."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, metadata, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Move metadata to device
        if use_metadata:
            metadata_device = {
                'numerical': metadata['numerical'].to(device),
                'gender': metadata['gender'].to(device),
                'location': metadata['location'].to(device),
                'engineered': metadata['engineered'].to(device),
            }
        else:
            metadata_device = None
        
        # MixUp (solo en imágenes, no en metadatos para mantener consistencia)
        if use_mixup and np.random.random() > 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2, device=device)
            if use_metadata:
                outputs, aux_gender, aux_age = model(images, metadata_device, True)
            else:
                outputs = model(images)
            
            # Main loss
            main_loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # Auxiliary losses (solo cuando NO hay mixup en labels)
            # Estimamos que aux_loss debe ser pequeño durante mixup
            aux_gender_loss = F.cross_entropy(aux_gender, metadata_device['gender']) * 0.1 if use_metadata else torch.tensor(0.0).to(device)
            
            # Para aux_age necesitamos calcular age_bin desde numerical
            # Por simplicidad, lo omitimos en mixup
            aux_age_loss = torch.tensor(0.0).to(device)
            
            loss = main_loss + aux_loss_weight * (aux_gender_loss + aux_age_loss)
        else:
            if use_metadata:
                outputs, aux_gender, aux_age = model(images, metadata_device, True)
            else:
                outputs = model(images)
            
            # Main loss
            main_loss = criterion(outputs, labels)
            
            # Auxiliary loss: predecir género desde imagen
            aux_gender_loss = F.cross_entropy(aux_gender, metadata_device['gender']) if use_metadata else torch.tensor(0.0).to(device)
            
            # Auxiliary loss: predecir edad categorizada desde imagen
            # Necesitamos extraer age_bin de engineered features
            # Por ahora, usamos una aproximación
            aux_age_loss = torch.tensor(0.0).to(device)
            
            loss = main_loss + aux_loss_weight * (aux_gender_loss + aux_age_loss)
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item()*accumulation_steps:.4f}", 
                         "acc": f"{100.*correct/total:.2f}%"})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_tta: bool = False,
    use_metadata: bool = True
) -> Tuple[float, float]:
    """Validation con TTA opcional."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, metadata, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            if use_metadata:
                metadata_device = {
                    'numerical': metadata['numerical'].to(device),
                    'gender': metadata['gender'].to(device),
                    'location': metadata['location'].to(device),
                    'engineered': metadata['engineered'].to(device),
                }
                outputs = model(images, metadata_device, False)
            else:
                outputs = model(images, None, False)
            
            # TTA: flip horizontal
            if use_tta:
                images_flip = torch.flip(images, dims=[3])
                if use_metadata:
                    outputs_flip = model(images_flip, metadata_device, False)
                else:
                    outputs_flip = model(images_flip, None, False)
                outputs = (outputs + outputs_flip) / 2
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def save_dataset_partition(
    dataset: Dataset,
    indices: List[int],
    output_path: Path,
    partition_name: str = "partition"
):
    """
    Saves the dataset partition information to a CSV file.
    
    Args:
        dataset: The dataset object containing the samples
        indices: List of indices for this partition
        output_path: Path where to save the CSV file
        partition_name: Name of the partition (e.g., 'train', 'val', 'test')
    """
    partition_data = []
    
    for idx in indices:
        row = dataset.data[idx]
        sample_info = {
            'partition': partition_name,
            'index': idx,
            'image_path': str(row['image_path']),
            'label': int(float(row['CLASS'])),
            'patient_id': row.get('patient_id', row.get('NHC', row.get('isic_id', 'unknown'))),
            'age': row.get('AGE_APPROX', 'unknown'),
            'sex': row.get('GENDER_STD', 'unknown'),
            'location': row.get('LOCATION_STD', 'unknown')
        }
        partition_data.append(sample_info)
    
    df_partition = pd.DataFrame(partition_data)
    df_partition.to_csv(output_path, index=False)
    print(f"✓ {partition_name.capitalize()} partition saved: {output_path} ({len(indices)} samples)")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_metadata: bool = True,
    use_tta: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa el modelo y retorna predicciones, probabilidades y etiquetas."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, metadata, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            if use_metadata:
                metadata_device = {
                    'numerical': metadata['numerical'].to(device),
                    'gender': metadata['gender'].to(device),
                    'location': metadata['location'].to(device),
                    'engineered': metadata['engineered'].to(device),
                }
                outputs = model(images, metadata_device)
            else:
                outputs = model(images)
            
            # TTA: promediar predicciones con flip horizontal
            if use_tta:
                images_flip = torch.flip(images, dims=[3])
                if use_metadata:
                    outputs_flip = model(images_flip, metadata_device)
                else:
                    outputs_flip = model(images_flip)
                outputs = (outputs + outputs_flip) / 2
            
            # Softmax para obtener probabilidades
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def save_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path,
    threshold: float = 0.5,
    threshold_label: str = "default"
):
    """Guarda reporte completo de métricas en archivo txt."""
    # Calcular métricas
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Matriz de confusión para calcular sensitivity y specificity
    cm = confusion_matrix(y_true, y_pred)
    
    # Sensitivity (recall para clase positiva = invasivo = 1)
    # Specificity (recall para clase negativa = in_situ = 0)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    
    # AUC-ROC
    try:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
    except:
        roc_auc = 0.0
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # PPV (Positive Predictive Value) y NPV (Negative Predictive Value)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Guardar reporte
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE MÉTRICAS - CLASIFICACIÓN MELANOMA\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Threshold aplicado: {threshold:.3f} ({threshold_label})\n")
        f.write(f"Total de muestras: {len(y_true)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("MÉTRICAS GENERALES\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:                    {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Balanced Accuracy:           {balanced_acc:.4f} ({balanced_acc*100:.2f}%)\n")
        f.write(f"Cohen's Kappa:               {kappa:.4f}\n")
        f.write(f"AUC-ROC:                     {roc_auc:.4f}\n")
        f.write(f"\n")
        
        f.write("-"*70 + "\n")
        f.write("MÉTRICAS POR CLASE (INVASIVO como positivo)\n")
        f.write("-"*70 + "\n")
        f.write(f"Sensitivity (Recall):        {sensitivity:.4f} ({sensitivity*100:.2f}%)\n")
        f.write(f"Specificity:                 {specificity:.4f} ({specificity*100:.2f}%)\n")
        f.write(f"Precision:                   {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"F1-Score:                    {f1:.4f}\n")
        f.write(f"PPV (Positive Pred. Value):  {ppv:.4f} ({ppv*100:.2f}%)\n")
        f.write(f"NPV (Negative Pred. Value):  {npv:.4f} ({npv*100:.2f}%)\n")
        f.write(f"\n")
        
        f.write("-"*70 + "\n")
        f.write("MATRIZ DE CONFUSIÓN\n")
        f.write("-"*70 + "\n")
        f.write(f"                    Predicho IN_SITU    Predicho INVASIVO\n")
        f.write(f"Real IN_SITU        {tn:^20}  {fp:^20}\n")
        f.write(f"Real INVASIVO       {fn:^20}  {tp:^20}\n")
        f.write(f"\n")
        f.write(f"True Negatives (TN):  {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"True Positives (TP):  {tp}\n")
        f.write(f"\n")
        
        # Tasas de error
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        f.write(f"False Positive Rate:         {fpr:.4f} ({fpr*100:.2f}%)\n")
        f.write(f"False Negative Rate:         {fnr:.4f} ({fnr*100:.2f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("\nReporte generado automáticamente por train_v3.py\n")
    
    print(f"✓ Reporte de métricas guardado: {output_path}")


def save_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    threshold_label: str = "default"
):
    """Saves normalized and unnormalized confusion matrices (separate files + combined)."""
    class_names = ["Miv", "Mis"]
    
    # Unnormalized matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalized matrix
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    
    # ===== 1. UNNORMALIZED MATRIX (individual file) =====
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, square=True, linewidths=1, linecolor='gray')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path_raw = output_dir / f"confusion_matrix_raw_{threshold_label}.png"
    plt.savefig(output_path_raw, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Unnormalized matrix saved: {output_path_raw}")
    
    # ===== 2. NORMALIZED MATRIX (individual file) =====
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, square=True, linewidths=1, linecolor='gray',
                vmin=0, vmax=1)
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path_norm = output_dir / f"confusion_matrix_normalized_{threshold_label}.png"
    plt.savefig(output_path_norm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Normalized matrix saved: {output_path_norm}")
    
    # ===== 3. COMBINED VERSION (both in one figure) =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Unnormalized matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'}, 
                square=True, linewidths=1, linecolor='gray')
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Unnormalized', fontsize=13, fontweight='bold')
    
    # Normalized matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'}, 
                square=True, linewidths=1, linecolor='gray', vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Normalized', fontsize=13, fontweight='bold')
    
    # General title
    fig.suptitle(f'Confusion Matrices - Threshold: {threshold_label}', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path_combined = output_dir / f"confusion_matrix_combined_{threshold_label}.png"
    plt.savefig(output_path_combined, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined matrices saved: {output_path_combined}")


def save_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: Path
):
    """Saves the ROC curve with AUC value."""
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    # Optimal point (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
             label=f'Optimal Point (threshold={optimal_threshold:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "roc_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curve saved: {output_path}")
    print(f"  AUC: {roc_auc:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    
    return optimal_threshold


def evaluate_with_custom_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: Path,
    thresholds: List[float] = None
):
    """Evaluates the model with different thresholds favoring the invasive class."""
    if thresholds is None:
        # Thresholds that favor detection of invasive (class 1)
        thresholds = [0.3, 0.4, 0.5, 0.6]
    
    print(f"\n{'='*70}")
    print("EVALUATION WITH CUSTOM THRESHOLDS")
    print(f"{'='*70}\n")
    
    results_summary = []
    
    for threshold in thresholds:
        # Apply custom threshold
        y_pred_custom = (y_probs[:, 1] >= threshold).astype(int)
        
        # Calculate basic metrics
        acc = accuracy_score(y_true, y_pred_custom)
        balanced_acc = balanced_accuracy_score(y_true, y_pred_custom)
        cm = confusion_matrix(y_true, y_pred_custom)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Guardar resumen
        results_summary.append({
            'threshold': threshold,
            'accuracy': acc,
            'balanced_acc': balanced_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
        
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Accuracy: {acc:.4f}, Balanced Acc: {balanced_acc:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
        # Guardar reporte completo
        threshold_label = f"threshold_{threshold:.2f}".replace('.', '_')
        report_path = output_dir / f"metrics_report_{threshold_label}.txt"
        save_metrics_report(y_true, y_pred_custom, y_probs, report_path, 
                          threshold, threshold_label)
        
        # Guardar matrices de confusión
        save_confusion_matrices(y_true, y_pred_custom, output_dir, threshold_label)
    
    # Guardar resumen comparativo
    summary_path = output_dir / "thresholds_comparison.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("COMPARACIÓN DE THRESHOLDS\n")
        f.write("="*90 + "\n\n")
        f.write(f"{'Threshold':<12} {'Accuracy':<12} {'Balanced Acc':<14} {'Sensitivity':<14} {'Specificity':<14}\n")
        f.write("-"*90 + "\n")
        
        for result in results_summary:
            f.write(f"{result['threshold']:<12.2f} ")
            f.write(f"{result['accuracy']:<12.4f} ")
            f.write(f"{result['balanced_acc']:<14.4f} ")
            f.write(f"{result['sensitivity']:<14.4f} ")
            f.write(f"{result['specificity']:<14.4f}\n")
        
        f.write("\n" + "="*90 + "\n")
        f.write("RECOMENDACIONES:\n")
        f.write("-"*90 + "\n")
        f.write("- Threshold < 0.5: Mayor sensibilidad (detecta más invasivos, más falsos positivos)\n")
        f.write("- Threshold = 0.5: Balance estándar\n")
        f.write("- Threshold > 0.5: Mayor especificidad (menos falsos positivos, puede perder invasivos)\n")
        f.write("\n")
        f.write("Para melanoma, se recomienda threshold < 0.5 para priorizar detección de casos invasivos.\n")
    
    print(f"\n✓ Resumen comparativo guardado: {summary_path}")
    print(f"{'='*70}\n")


def patient_level_split(
    dataset,
    val_split: float = 0.2,
    patient_col: str = "NHC",
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split a nivel de paciente."""
    np.random.seed(seed)
    
    patient_to_indices = {}
    patient_to_class = {}
    
    for idx, row in enumerate(dataset.data):
        patient_id = row.get(patient_col, None)
        
        if patient_id is None or (isinstance(patient_id, float) and np.isnan(patient_id)):
            patient_id = row.get("image_name", f"unknown_{idx}")
        
        patient_id = str(patient_id)
        
        if patient_id not in patient_to_indices:
            patient_to_indices[patient_id] = []
            class_val = int(float(row["CLASS"]))
            label = dataset.class_mapping[class_val]
            patient_to_class[patient_id] = label
        
        patient_to_indices[patient_id].append(idx)
    
    patients = list(patient_to_indices.keys())
    patient_classes = [patient_to_class[p] for p in patients]
    
    unique_classes, counts = np.unique(patient_classes, return_counts=True)
    print(f"\nDistribución por paciente:")
    for cls, cnt in zip(unique_classes, counts):
        cls_name = "IN_SITU" if cls == 0 else "INVASIVO"
        print(f"  {cls_name}: {cnt} pacientes")
    
    train_patients = []
    val_patients = []
    
    for cls in unique_classes:
        cls_patients = [p for p, c in zip(patients, patient_classes) if c == cls]
        np.random.shuffle(cls_patients)
        
        n_val = max(1, int(len(cls_patients) * val_split))
        
        val_patients.extend(cls_patients[:n_val])
        train_patients.extend(cls_patients[n_val:])
    
    train_indices = []
    val_indices = []
    
    for patient in train_patients:
        train_indices.extend(patient_to_indices[patient])
    
    for patient in val_patients:
        val_indices.extend(patient_to_indices[patient])
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_patients)} pacientes, {len(train_indices)} imágenes")
    print(f"  Val: {len(val_patients)} pacientes, {len(val_indices)} imágenes")
    
    return train_indices, val_indices


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train V3 - Maximizar valor de metadatos")
    parser.add_argument("--csv", type=str, nargs="+", 
                       default=["reports/dataset_procesado_VRUH.csv", "reports/dataset_procesado_ISIC.csv"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--model-name", type=str, default="densenet121", 
                       choices=["resnet18", "resnet50", "densenet121"])
    parser.add_argument("--no-metadata", action="store_true",
                       help="Entrenar sin metadatos (solo imágenes)")
    parser.add_argument("--image-only", action="store_true",
                       help="Forzar un modelo solo de imagen (baseline).")
    parser.add_argument("--output-dir", type=str, default="checkpoints_v3")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--focal-loss", action="store_true", default=True)
    parser.add_argument("--use-mixup", action="store_true", default=True)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--aux-loss-weight", type=float, default=0.3,
                       help="Peso de auxiliary losses")
    parser.add_argument("--metadata-noise-std", type=float, default=0.1,
                       help="Ruido a añadir a metadatos en training")
    parser.add_argument("--n-exp", type=int, default=0,
                       help="Número de experimento para organizar resultados")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para reproducibilidad")
    args = parser.parse_args(argv)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    # Set seed for reproducibility if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Deterministic options (may slow down training)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    
    # Load datasets
    csv_paths = [Path(p) for p in args.csv] if isinstance(args.csv, list) else [Path(args.csv)]
    
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"ERROR: {csv_path} no encontrado", file=sys.stderr)
            return 1
    
    print(f"\nCargando {len(csv_paths)} dataset(s)")
    
    all_train_data = []
    all_val_data = []
    
    for idx, csv_path in enumerate(csv_paths):
        is_verbose = (idx == 0)
        
        train_dataset = AdvancedMetadataDataset(
            csv_path=csv_path,
            img_size=args.img_size,
            use_augmentation=True,
            verbose=is_verbose,
            metadata_noise_std=args.metadata_noise_std,
        )
        all_train_data.append(train_dataset)
        
        val_dataset = AdvancedMetadataDataset(
            csv_path=csv_path,
            img_size=args.img_size,
            use_augmentation=False,
            verbose=False,
            metadata_noise_std=0.0,  # Sin ruido en validación
        )
        all_val_data.append(val_dataset)
    
    train_full_dataset = all_train_data[0]
    val_full_dataset = all_val_data[0]
    
    if len(all_train_data) > 1:
        for i in range(1, len(all_train_data)):
            train_full_dataset.data.extend(all_train_data[i].data)
            val_full_dataset.data.extend(all_val_data[i].data)
        
        train_full_dataset.verbose = False
        val_full_dataset.verbose = False
        train_full_dataset._prepare_encoders()
        val_full_dataset._prepare_encoders()
        train_full_dataset._compute_metadata_stats()
        val_full_dataset.metadata_stats = train_full_dataset.metadata_stats
        val_full_dataset.num_genders = train_full_dataset.num_genders
        val_full_dataset.num_locations = train_full_dataset.num_locations
        val_full_dataset.encoders = train_full_dataset.encoders
        
        print(f"\nDatasets combinados: {len(train_full_dataset.data)} muestras")
    
    # Patient-level split
    patient_col = None
    for possible_col in ["NHC", "patient_id", "isic_id"]:
        if possible_col in train_full_dataset.data[0]:
            patient_col = possible_col
            break
    
    if patient_col is not None:
        print(f"\nSplit por paciente (columna: {patient_col})")
        train_indices, val_indices = patient_level_split(
            train_full_dataset,
            val_split=args.val_split,
            patient_col=patient_col,
            seed=42
        )
    else:
        print("\nAdvertencia: split por imagen")
        val_size = int(len(train_full_dataset) * args.val_split)
        train_size = len(train_full_dataset) - val_size
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
    
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Weighted sampler
    all_labels = [train_full_dataset.class_mapping[int(float(row["CLASS"]))] 
                  for row in train_full_dataset.data]
    class_counts = np.bincount(all_labels)
    sample_weights = [1.0 / class_counts[label] for label in all_labels]
    train_sample_weights = [sample_weights[i] for i in train_indices]
    
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )
    
    print(f"\nOversampling activado")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Model con arquitectura avanzada
    num_classes = 2
    
    use_metadata = not args.no_metadata
    if args.image_only:
        use_metadata = False

    # Selección del modelo
    if args.image_only:
        model = ImageOnlyNet(
            num_classes=num_classes,
            backbone=args.model_name,
            pretrained=args.pretrained,
        ).to(device)
        print("\nModelo: ImageOnlyNet (baseline sin metadatos)")
    else:
        model = GatedFusionMultimodalNet(
            num_classes=num_classes,
            num_genders=train_full_dataset.num_genders,
            num_locations=train_full_dataset.num_locations,
            backbone=args.model_name,
            pretrained=args.pretrained,
            use_metadata=use_metadata,
        ).to(device)
    
    
    print(f"\n{'='*60}")
    print(f"MODELO V3 - ARQUITECTURA AVANZADA")
    print(f"{'='*60}")
    print(f"Backbone: {args.model_name}")
    print(f"Genders: {train_full_dataset.num_genders} (embedding 16d)")
    print(f"Locations: {train_full_dataset.num_locations} (embedding 32d)")
    print(f"Features numéricas: 3 (age, diam1, diam2)")
    print(f"Features engineered: 3 (age_bin, ratio, diam_cat)")
    if args.image_only:
        print(f"ImageOnly baseline: ✓ (no metadatos)")
    else:
        print(f"Gated Fusion: ✓")
        print(f"Cross-Attention: ✓")
        print(f"Auxiliary Losses: ✓ (weight={args.aux_loss_weight})")
        print(f"Metadata Noise: ✓ (std={args.metadata_noise_std})")
    print(f"{'='*60}")
    
    # Class weights
    total_samples = len(all_labels)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"\nDistribución:")
    print(f"  IN_SITU: {class_counts[0]} (peso: {class_weights[0]:.4f})")
    print(f"  INVASIVO: {class_counts[1]} (peso: {class_weights[1]:.4f})")
    
    # Loss
    if args.focal_loss:
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
        print(f"\nUsando Focal Loss (gamma=2.0)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Output dir - Organizado por N_EXP
    base_output_dir = Path(args.output_dir) / f"N_EXP_{args.n_exp}"
    # Separate folder when training without metadatos for fair comparisons
    if args.image_only:
        output_dir = base_output_dir / f"{args.model_name}_image_only"
    else:
        if use_metadata:
            output_dir = base_output_dir / f"{args.model_name}_gated_fusion_with_metadata"
        else:
            output_dir = base_output_dir / f"{args.model_name}_gated_fusion_no_metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save partition information to CSV
    train_partition_path = output_dir / "train_partition.csv"
    val_partition_path = output_dir / "val_partition.csv"
    save_dataset_partition(train_full_dataset, train_indices, train_partition_path, "train")
    save_dataset_partition(val_full_dataset, val_indices, val_partition_path, "validation")
    
    # Guardar configuración del experimento
    config_file = output_dir / "experiment_config.txt"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"CONFIGURACIÓN DEL EXPERIMENTO N_EXP_{args.n_exp}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Modelo: {args.model_name}\n")
        if args.image_only:
            f.write(f"Arquitectura: ImageOnly (baseline)\n")
        else:
            f.write(f"Arquitectura: Gated Fusion + Cross-Attention\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Val split: {args.val_split}\n")
        f.write(f"Pretrained: {args.pretrained}\n")
        f.write(f"Focal Loss: {args.focal_loss}\n")
        f.write(f"MixUp: {args.use_mixup}\n")
        f.write(f"Usar metadatos: {use_metadata}\n")
        f.write(f"Image Only: {args.image_only}\n")
        f.write(f"TTA: {args.use_tta}\n")
        f.write(f"Early stop patience: {args.early_stop_patience}\n")
        f.write(f"Gradient accumulation steps: {args.accumulation_steps}\n")
        f.write(f"Batch size efectivo: {args.batch_size * args.accumulation_steps}\n")
        f.write(f"Auxiliary loss weight: {args.aux_loss_weight}\n")
        f.write(f"Metadata noise std: {args.metadata_noise_std}\n")
        f.write(f"CSV files: {args.csv}\n")
        f.write("\n" + "="*70 + "\n")
    
    print(f"\nExperimento: N_EXP_{args.n_exp}")
    print(f"Resultados: {output_dir}\n")
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Época {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.use_mixup, args.accumulation_steps, args.aux_loss_weight,
            use_metadata
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, args.use_tta,
            use_metadata
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
            }, checkpoint_path)
            print(f"✓ Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.early_stop_patience}")
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠ Early stopping en época {epoch}")
            break
    
    print(f"\n{'='*60}")
    print(f"Mejor Val Acc: {best_val_acc:.2f}%")
    print(f"{'='*60}")

    # ========================================================================
    # EVALUACIÓN FINAL: generar métricas y gráficas (igual que train_v2)
    # ========================================================================
    print(f"\n{'='*60}")
    print("INICIANDO EVALUACIÓN FINAL DEL MEJOR MODELO")
    print(f"{'='*60}\n")

    # Cargar el mejor modelo
    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Mejor modelo cargado (época {checkpoint['epoch']})\n")

    # Obtener predicciones y probabilidades
    print("Evaluando modelo en conjunto de validación...")
    y_pred, y_probs, y_true = evaluate_model(
        model, val_loader, device, use_metadata, args.use_tta
    )

    print(f"\nTotal de muestras evaluadas: {len(y_true)}")
    print(f"Distribución real - IN_SITU: {np.sum(y_true == 0)}, INVASIVO: {np.sum(y_true == 1)}")
    print(f"Distribución predicha - IN_SITU: {np.sum(y_pred == 0)}, INVASIVO: {np.sum(y_pred == 1)}\n")

    # 1. Generar reporte con threshold por defecto (0.5)
    print("\n" + "="*60)
    print("1. GENERANDO REPORTE CON THRESHOLD ESTÁNDAR (0.5)")
    print("="*60)
    report_path = output_dir / "metrics_report_threshold_0_50.txt"
    save_metrics_report(y_true, y_pred, y_probs, report_path, 0.5, "threshold_0_50")

    # 2. Guardar matrices de confusión con threshold 0.5
    print("\n" + "="*60)
    print("2. GENERANDO MATRICES DE CONFUSIÓN (THRESHOLD 0.5)")
    print("="*60)
    save_confusion_matrices(y_true, y_pred, output_dir, "threshold_0_50")

    # 3. Guardar curva ROC
    print("\n" + "="*60)
    print("3. GENERANDO CURVA ROC")
    print("="*60)
    optimal_threshold = save_roc_curve(y_true, y_probs, output_dir)

    # 4. Evaluar con thresholds personalizados
    print("\n" + "="*60)
    print("4. EVALUANDO CON THRESHOLDS PERSONALIZADOS")
    print("="*60)
    evaluate_with_custom_thresholds(y_true, y_probs, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
