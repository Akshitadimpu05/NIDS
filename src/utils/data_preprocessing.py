"""
Data preprocessing utilities for NIDS training and inference.
Handles CIC-Darknet2020 dataset and real-time traffic data preprocessing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pickle
import os

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessor for NIDS datasets.
    Handles feature engineering, normalization, and data splitting.
    """
    
    def __init__(self, 
                 normalization_method: str = 'standard',
                 handle_missing: str = 'mean',
                 feature_selection: bool = True,
                 remove_duplicates: bool = True):
        """
        Initialize data preprocessor.
        
        Args:
            normalization_method: Normalization method ('standard', 'minmax', 'robust')
            handle_missing: How to handle missing values ('mean', 'median', 'drop')
            feature_selection: Whether to perform feature selection
            remove_duplicates: Whether to remove duplicate samples
        """
        self.normalization_method = normalization_method
        self.handle_missing = handle_missing
        self.feature_selection = feature_selection
        self.remove_duplicates = remove_duplicates
        
        # Preprocessing components
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = None
        self.selected_features = None
        
        # Statistics
        self.preprocessing_stats = {}
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize preprocessing components."""
        # Scaler
        if self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalization_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        # Imputer
        if self.handle_missing in ['mean', 'median']:
            self.imputer = SimpleImputer(strategy=self.handle_missing)
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        logger.info(f"Data preprocessor initialized with {self.normalization_method} normalization")
    
    def load_cic_darknet2020(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess CIC-Darknet2020 dataset.
        
        Args:
            data_path: Path to the dataset file or directory
            
        Returns:
            Tuple of (features_df, labels_df)
        """
        logger.info(f"Loading CIC-Darknet2020 dataset from {data_path}")
        
        csv_files = []
        
        # Check if data_path is a file or directory
        if os.path.isfile(data_path):
            # Single CSV file
            if data_path.endswith('.csv') or data_path.endswith('.CSV'):
                csv_files.append(data_path)
            else:
                raise ValueError(f"File {data_path} is not a CSV file")
        elif os.path.isdir(data_path):
            # Directory - find CSV files in the directory
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
        else:
            raise FileNotFoundError(f"Path {data_path} does not exist")
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        logger.info(f"Found {len(csv_files)} CSV file(s) to load")
        
        # Load and combine all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                # Load CSV with error handling for malformed rows
                logger.info(f"Loading {csv_file}...")
                df = pd.read_csv(csv_file, low_memory=False, on_bad_lines='skip')
                
                # Clean up duplicate columns (like Label,Label)
                if 'Label' in df.columns:
                    # Keep only the first Label column if there are duplicates
                    label_cols = [col for col in df.columns if col == 'Label']
                    if len(label_cols) > 1:
                        # Drop duplicate label columns except the first
                        cols_to_drop = label_cols[1:]
                        df = df.drop(columns=cols_to_drop)
                        logger.info(f"Removed {len(cols_to_drop)} duplicate Label columns")
                
                dataframes.append(df)
                logger.info(f"Successfully loaded {len(df)} samples from {csv_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_df.shape}")
        
        # Separate features and labels
        label_column = self._identify_label_column(combined_df)
        
        if label_column:
            features_df = combined_df.drop(columns=[label_column])
            labels_df = combined_df[label_column]
        else:
            # If no clear label column, assume last column is label
            features_df = combined_df.iloc[:, :-1]
            labels_df = combined_df.iloc[:, -1]
        
        # Filter out non-numeric columns (Flow ID, IPs, timestamps, etc.)
        numeric_columns = []
        non_numeric_columns = []
        
        for col in features_df.columns:
            try:
                # Try to convert to numeric
                pd.to_numeric(features_df[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                non_numeric_columns.append(col)
        
        if non_numeric_columns:
            logger.info(f"Removing {len(non_numeric_columns)} non-numeric columns: {non_numeric_columns[:5]}...")
            features_df = features_df[numeric_columns]
        
        logger.info(f"Features shape: {features_df.shape}, Labels shape: {labels_df.shape}")
        logger.info(f"Using {len(numeric_columns)} numeric features for training")
        
        return features_df, labels_df
    
    def _identify_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the label column in the dataset."""
        # Common label column names
        label_candidates = [
            'Label', 'label', 'class', 'Class', 'target', 'Target',
            'attack', 'Attack', 'category', 'Category'
        ]
        
        for col in label_candidates:
            if col in df.columns:
                return col
        
        # Look for columns with string values (likely labels)
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_values = df[col].nunique()
                if unique_values < 20:  # Reasonable number of classes
                    return col
        
        return None
    
    def preprocess_features(self, 
                           features: Union[pd.DataFrame, np.ndarray],
                           fit: bool = True) -> np.ndarray:
        """
        Preprocess feature data.
        
        Args:
            features: Feature data
            fit: Whether to fit preprocessing components
            
        Returns:
            Preprocessed feature array
        """
        if isinstance(features, pd.DataFrame):
            if fit:
                self.feature_names = list(features.columns)
            features_array = features.values
        else:
            features_array = features
        
        logger.info(f"Preprocessing features with shape: {features_array.shape}")
        
        # Clean data: handle infinite values, NaN, and extremely large numbers
        logger.info("Cleaning data - handling infinite values and outliers...")
        
        # Check for infinite and NaN values
        inf_count = np.isinf(features_array).sum()
        nan_count = np.isnan(features_array).sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values, replacing...")
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, replacing...")
        
        # Replace infinite and NaN values
        features_array = np.nan_to_num(features_array, 
                                     nan=0.0, 
                                     posinf=1e10,  # Use smaller max value
                                     neginf=-1e10)
        
        # Clip extremely large values to prevent overflow
        features_array = np.clip(features_array, -1e15, 1e15)
        
        # Check for any remaining problematic values
        if np.any(np.isinf(features_array)) or np.any(np.isnan(features_array)):
            logger.error("Still have infinite or NaN values after cleaning!")
            # Force replace any remaining problematic values
            features_array[np.isinf(features_array)] = 0.0
            features_array[np.isnan(features_array)] = 0.0
        
        logger.info("Data cleaning completed")
        
        # Handle missing values with imputer
        if self.imputer and fit:
            features_array = self.imputer.fit_transform(features_array)
        elif self.imputer:
            features_array = self.imputer.transform(features_array)
        
        # Feature selection
        if self.feature_selection and fit:
            self.selected_features = self._select_features(features_array)
        
        if self.selected_features is not None:
            features_array = features_array[:, self.selected_features]
        
        # Normalization
        if fit:
            features_array = self.scaler.fit_transform(features_array)
        else:
            features_array = self.scaler.transform(features_array)
        
        logger.info(f"Preprocessed features shape: {features_array.shape}")
        
        return features_array.astype(np.float32)
    
    def preprocess_labels(self, 
                         labels: Union[pd.Series, np.ndarray],
                         fit: bool = True) -> np.ndarray:
        """
        Preprocess label data.
        
        Args:
            labels: Label data
            fit: Whether to fit label encoder
            
        Returns:
            Encoded label array
        """
        if isinstance(labels, pd.Series):
            labels_array = labels.values
        else:
            labels_array = labels
        
        logger.info(f"Preprocessing labels with shape: {labels_array.shape}")
        
        # Handle string labels
        if labels_array.dtype == 'object' or isinstance(labels_array[0], str):
            if fit:
                encoded_labels = self.label_encoder.fit_transform(labels_array)
                logger.info(f"Label classes: {list(self.label_encoder.classes_)}")
            else:
                encoded_labels = self.label_encoder.transform(labels_array)
        else:
            encoded_labels = labels_array.astype(np.int64)
        
        return encoded_labels
    
    def _select_features(self, features: np.ndarray) -> np.ndarray:
        """
        Select most relevant features.
        
        Args:
            features: Feature array
            
        Returns:
            Indices of selected features
        """
        # Remove features with zero variance
        variances = np.var(features, axis=0)
        non_zero_var = variances > 1e-8
        
        # Remove highly correlated features
        if features.shape[1] > 1:
            correlation_matrix = np.corrcoef(features.T)
            correlation_matrix = np.nan_to_num(correlation_matrix)
            
            # Find highly correlated pairs
            high_corr_pairs = np.where(np.abs(correlation_matrix) > 0.95)
            features_to_remove = set()
            
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i != j and i not in features_to_remove:
                    features_to_remove.add(j)
            
            # Create mask for features to keep
            keep_features = np.ones(features.shape[1], dtype=bool)
            keep_features[list(features_to_remove)] = False
            keep_features = keep_features & non_zero_var
        else:
            keep_features = non_zero_var
        
        selected_indices = np.where(keep_features)[0]
        
        logger.info(f"Selected {len(selected_indices)} features out of {features.shape[1]}")
        
        return selected_indices
    
    def create_train_test_split(self, 
                               features: np.ndarray,
                               labels: np.ndarray,
                               test_size: float = 0.2,
                               validation_size: float = 0.1,
                               random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Create train/validation/test splits.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Proportion of test set
            validation_size: Proportion of validation set
            random_state: Random seed
            
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Creating train/validation/test splits")
        
        # Remove duplicates if requested
        if self.remove_duplicates:
            unique_indices = self._find_unique_samples(features)
            features = features[unique_indices]
            labels = labels[unique_indices]
            logger.info(f"Removed duplicates, remaining samples: {len(features)}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        # Log split information
        for split_name, split_data in splits.items():
            logger.info(f"{split_name} shape: {split_data.shape}")
        
        # Store preprocessing statistics
        self.preprocessing_stats = {
            'total_samples': len(features),
            'num_features': features.shape[1],
            'num_classes': len(np.unique(labels)),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'class_distribution': {
                str(cls): int(count) for cls, count in 
                zip(*np.unique(labels, return_counts=True))
            }
        }
        
        return splits
    
    def _find_unique_samples(self, features: np.ndarray) -> np.ndarray:
        """Find unique samples in the dataset."""
        # Use a hash-based approach for efficiency
        feature_hashes = []
        for i in range(features.shape[0]):
            # Create hash of the feature vector
            feature_hash = hash(tuple(features[i]))
            feature_hashes.append(feature_hash)
        
        # Find unique hashes
        unique_hashes, unique_indices = np.unique(feature_hashes, return_index=True)
        
        return unique_indices
    
    def preprocess_realtime_data(self, traffic_features: List[float]) -> np.ndarray:
        """
        Preprocess real-time traffic data for inference.
        
        Args:
            traffic_features: List of traffic features
            
        Returns:
            Preprocessed feature array ready for model inference
        """
        # Convert to numpy array
        features = np.array(traffic_features, dtype=np.float32).reshape(1, -1)
        
        # Handle missing values
        if self.imputer:
            features = self.imputer.transform(features)
        
        # Remove infinite values
        features = np.nan_to_num(features,
                                nan=0.0,
                                posinf=np.finfo(np.float32).max,
                                neginf=np.finfo(np.float32).min)
        
        # Feature selection
        if self.selected_features is not None:
            features = features[:, self.selected_features]
        
        # Normalization
        if self.scaler:
            features = self.scaler.transform(features)
        
        return features.astype(np.float32)
    
    def create_anomaly_detection_data(self, 
                                    features: np.ndarray,
                                    labels: np.ndarray,
                                    normal_class: Union[str, int] = 0) -> Dict[str, np.ndarray]:
        """
        Create dataset for anomaly detection (normal vs. anomaly).
        
        Args:
            features: Feature array
            labels: Label array
            normal_class: Label value representing normal traffic
            
        Returns:
            Dictionary with normal and anomaly data
        """
        logger.info("Creating anomaly detection dataset")
        
        # Identify normal and anomaly samples
        normal_mask = labels == normal_class
        anomaly_mask = ~normal_mask
        
        normal_features = features[normal_mask]
        anomaly_features = features[anomaly_mask]
        
        logger.info(f"Normal samples: {len(normal_features)}")
        logger.info(f"Anomaly samples: {len(anomaly_features)}")
        
        return {
            'normal_features': normal_features,
            'anomaly_features': anomaly_features,
            'normal_labels': np.zeros(len(normal_features)),
            'anomaly_labels': np.ones(len(anomaly_features))
        }
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor components to file."""
        preprocessor_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'preprocessing_stats': self.preprocessing_stats,
            'config': {
                'normalization_method': self.normalization_method,
                'handle_missing': self.handle_missing,
                'feature_selection': self.feature_selection,
                'remove_duplicates': self.remove_duplicates
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor components from file."""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.imputer = preprocessor_data['imputer']
        self.label_encoder = preprocessor_data['label_encoder']
        self.feature_names = preprocessor_data['feature_names']
        self.selected_features = preprocessor_data['selected_features']
        self.preprocessing_stats = preprocessor_data['preprocessing_stats']
        
        # Update config
        config = preprocessor_data.get('config', {})
        self.normalization_method = config.get('normalization_method', self.normalization_method)
        self.handle_missing = config.get('handle_missing', self.handle_missing)
        self.feature_selection = config.get('feature_selection', self.feature_selection)
        self.remove_duplicates = config.get('remove_duplicates', self.remove_duplicates)
        
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def get_feature_importance(self, model_feature_importance: np.ndarray) -> Dict[str, float]:
        """
        Get feature importance mapping.
        
        Args:
            model_feature_importance: Feature importance from trained model
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_names is None:
            return {}
        
        # Map importance to original feature names
        if self.selected_features is not None:
            feature_names = [self.feature_names[i] for i in self.selected_features]
        else:
            feature_names = self.feature_names
        
        if len(feature_names) != len(model_feature_importance):
            logger.warning("Feature names and importance array length mismatch")
            return {}
        
        return dict(zip(feature_names, model_feature_importance))
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations."""
        return {
            'config': {
                'normalization_method': self.normalization_method,
                'handle_missing': self.handle_missing,
                'feature_selection': self.feature_selection,
                'remove_duplicates': self.remove_duplicates
            },
            'statistics': self.preprocessing_stats,
            'components': {
                'scaler_fitted': self.scaler is not None,
                'imputer_fitted': self.imputer is not None,
                'label_encoder_fitted': self.label_encoder is not None,
                'features_selected': self.selected_features is not None
            },
            'feature_info': {
                'total_features': len(self.feature_names) if self.feature_names else 0,
                'selected_features': len(self.selected_features) if self.selected_features is not None else 0,
                'feature_names': self.feature_names[:10] if self.feature_names else []  # First 10 for brevity
            }
        }
