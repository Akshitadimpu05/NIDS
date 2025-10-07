"""
Model Aggregator for Federated Learning in NIDS.
Implements federated averaging and other aggregation strategies.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import copy

from ..models.ensemble import HybridNIDSModel

logger = logging.getLogger(__name__)


class ModelAggregator:
    """
    Federated learning model aggregator for NIDS.
    Implements various aggregation strategies for combining models from fog nodes.
    """
    
    def __init__(self, 
                 aggregation_strategy: str = 'federated_averaging',
                 min_nodes_for_aggregation: int = 3,
                 weight_by_data_size: bool = True,
                 quality_threshold: float = 0.1):
        """
        Initialize model aggregator.
        
        Args:
            aggregation_strategy: Strategy for aggregation ('federated_averaging', 'weighted_average')
            min_nodes_for_aggregation: Minimum nodes required for aggregation
            weight_by_data_size: Whether to weight by training data size
            quality_threshold: Minimum quality threshold for including models
        """
        self.aggregation_strategy = aggregation_strategy
        self.min_nodes_for_aggregation = min_nodes_for_aggregation
        self.weight_by_data_size = weight_by_data_size
        self.quality_threshold = quality_threshold
        
        # Aggregation history
        self.aggregation_history = []
        
        logger.info(f"Model aggregator initialized with strategy: {aggregation_strategy}")
    
    async def aggregate_models(self, model_updates: List[Dict[str, Any]]) -> Optional[HybridNIDSModel]:
        """
        Aggregate models from multiple fog nodes.
        
        Args:
            model_updates: List of model updates from fog nodes
            
        Returns:
            Aggregated global model or None if aggregation failed
        """
        if len(model_updates) < self.min_nodes_for_aggregation:
            logger.warning(f"Insufficient models for aggregation: {len(model_updates)} < {self.min_nodes_for_aggregation}")
            return None
        
        logger.info(f"Aggregating {len(model_updates)} models using {self.aggregation_strategy}")
        
        try:
            # Filter models by quality
            quality_models = self._filter_models_by_quality(model_updates)
            
            if len(quality_models) < self.min_nodes_for_aggregation:
                logger.warning("Insufficient quality models after filtering")
                return None
            
            # Perform aggregation based on strategy
            if self.aggregation_strategy == 'federated_averaging':
                aggregated_model = await self._federated_averaging(quality_models)
            elif self.aggregation_strategy == 'weighted_average':
                aggregated_model = await self._weighted_averaging(quality_models)
            else:
                raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
            
            # Record aggregation
            self.aggregation_history.append({
                'timestamp': torch.tensor(0).item(),  # Current time
                'num_models': len(quality_models),
                'strategy': self.aggregation_strategy,
                'success': aggregated_model is not None
            })
            
            return aggregated_model
            
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
            return None
    
    def _filter_models_by_quality(self, model_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter models based on quality metrics."""
        quality_models = []
        
        for update in model_updates:
            model_data = update.get('model_data', {})
            
            # Check if model has quality metrics
            performance_metrics = model_data.get('performance_metrics', {})
            
            # Simple quality check based on processing performance
            if performance_metrics:
                avg_processing_time = performance_metrics.get('avg_processing_time', float('inf'))
                anomalies_detected = performance_metrics.get('anomalies_detected', 0)
                
                # Quality score based on efficiency and detection capability
                if avg_processing_time < 1.0 and anomalies_detected >= 0:  # Basic thresholds
                    quality_models.append(update)
                else:
                    logger.debug(f"Model filtered out due to quality: processing_time={avg_processing_time}")
            else:
                # Include models without metrics (assume they're valid)
                quality_models.append(update)
        
        logger.info(f"Quality filtering: {len(quality_models)}/{len(model_updates)} models passed")
        return quality_models
    
    async def _federated_averaging(self, model_updates: List[Dict[str, Any]]) -> Optional[HybridNIDSModel]:
        """
        Implement federated averaging algorithm.
        
        Args:
            model_updates: List of model updates
            
        Returns:
            Aggregated model
        """
        try:
            # Initialize aggregated model
            aggregated_model = HybridNIDSModel()
            
            # Calculate weights for each model
            weights = self._calculate_aggregation_weights(model_updates)
            
            # Aggregate each component separately
            aggregated_ae_state = self._aggregate_autoencoder_weights(model_updates, weights)
            aggregated_capsnet_state = self._aggregate_capsnet_weights(model_updates, weights)
            aggregated_rl_state = self._aggregate_rl_weights(model_updates, weights)
            
            # Load aggregated weights into model
            if aggregated_ae_state:
                aggregated_model.autoencoder.load_state_dict(aggregated_ae_state)
            
            if aggregated_capsnet_state:
                aggregated_model.capsnet.load_state_dict(aggregated_capsnet_state)
            
            if aggregated_rl_state:
                aggregated_model.rl_agent.network.load_state_dict(aggregated_rl_state)
            
            # Update training status
            aggregated_model.is_trained = {
                'autoencoder': aggregated_ae_state is not None,
                'capsnet': aggregated_capsnet_state is not None,
                'rl_agent': aggregated_rl_state is not None
            }
            
            logger.info("Federated averaging completed successfully")
            return aggregated_model
            
        except Exception as e:
            logger.error(f"Federated averaging failed: {e}")
            return None
    
    async def _weighted_averaging(self, model_updates: List[Dict[str, Any]]) -> Optional[HybridNIDSModel]:
        """
        Implement weighted averaging based on node performance.
        
        Args:
            model_updates: List of model updates
            
        Returns:
            Aggregated model
        """
        # For now, use the same implementation as federated averaging
        # In a more sophisticated version, this could use different weighting schemes
        return await self._federated_averaging(model_updates)
    
    def _calculate_aggregation_weights(self, model_updates: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate weights for model aggregation.
        
        Args:
            model_updates: List of model updates
            
        Returns:
            List of weights for each model
        """
        if not self.weight_by_data_size:
            # Equal weights
            return [1.0 / len(model_updates)] * len(model_updates)
        
        # Weight by data size or performance metrics
        weights = []
        total_weight = 0
        
        for update in model_updates:
            node_capabilities = update.get('node_capabilities', {})
            model_data = update.get('model_data', {})
            
            # Use processing capacity as weight proxy
            processing_threads = node_capabilities.get('processing_threads', 1)
            performance_metrics = model_data.get('performance_metrics', {})
            packets_processed = performance_metrics.get('packets_processed', 1)
            
            # Combine thread count and processing volume
            weight = processing_threads * np.log(packets_processed + 1)
            weights.append(weight)
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(model_updates)] * len(model_updates)
        
        logger.debug(f"Aggregation weights: {weights}")
        return weights
    
    def _aggregate_autoencoder_weights(self, 
                                     model_updates: List[Dict[str, Any]], 
                                     weights: List[float]) -> Optional[Dict[str, torch.Tensor]]:
        """Aggregate autoencoder weights."""
        try:
            # Extract autoencoder state dicts
            ae_states = []
            valid_weights = []
            
            for i, update in enumerate(model_updates):
                model_data = update.get('model_data', {})
                if 'autoencoder_state_dict' in model_data:
                    ae_states.append(model_data['autoencoder_state_dict'])
                    valid_weights.append(weights[i])
            
            if not ae_states:
                logger.warning("No autoencoder states found for aggregation")
                return None
            
            # Normalize weights for valid models
            total_weight = sum(valid_weights)
            if total_weight > 0:
                valid_weights = [w / total_weight for w in valid_weights]
            
            # Aggregate weights
            aggregated_state = {}
            
            # Get parameter names from first model
            param_names = ae_states[0].keys()
            
            for param_name in param_names:
                # Weighted average of parameters
                weighted_param = None
                
                for state_dict, weight in zip(ae_states, valid_weights):
                    if param_name in state_dict:
                        param_tensor = state_dict[param_name]
                        
                        if weighted_param is None:
                            weighted_param = weight * param_tensor
                        else:
                            weighted_param += weight * param_tensor
                
                if weighted_param is not None:
                    aggregated_state[param_name] = weighted_param
            
            logger.debug(f"Aggregated autoencoder with {len(ae_states)} models")
            return aggregated_state
            
        except Exception as e:
            logger.error(f"Autoencoder aggregation failed: {e}")
            return None
    
    def _aggregate_capsnet_weights(self, 
                                  model_updates: List[Dict[str, Any]], 
                                  weights: List[float]) -> Optional[Dict[str, torch.Tensor]]:
        """Aggregate CapsNet weights."""
        try:
            # Extract CapsNet state dicts
            capsnet_states = []
            valid_weights = []
            
            for i, update in enumerate(model_updates):
                model_data = update.get('model_data', {})
                if 'capsnet_state_dict' in model_data:
                    capsnet_states.append(model_data['capsnet_state_dict'])
                    valid_weights.append(weights[i])
            
            if not capsnet_states:
                logger.warning("No CapsNet states found for aggregation")
                return None
            
            # Normalize weights
            total_weight = sum(valid_weights)
            if total_weight > 0:
                valid_weights = [w / total_weight for w in valid_weights]
            
            # Aggregate weights
            aggregated_state = {}
            param_names = capsnet_states[0].keys()
            
            for param_name in param_names:
                weighted_param = None
                
                for state_dict, weight in zip(capsnet_states, valid_weights):
                    if param_name in state_dict:
                        param_tensor = state_dict[param_name]
                        
                        if weighted_param is None:
                            weighted_param = weight * param_tensor
                        else:
                            weighted_param += weight * param_tensor
                
                if weighted_param is not None:
                    aggregated_state[param_name] = weighted_param
            
            logger.debug(f"Aggregated CapsNet with {len(capsnet_states)} models")
            return aggregated_state
            
        except Exception as e:
            logger.error(f"CapsNet aggregation failed: {e}")
            return None
    
    def _aggregate_rl_weights(self, 
                             model_updates: List[Dict[str, Any]], 
                             weights: List[float]) -> Optional[Dict[str, torch.Tensor]]:
        """Aggregate RL agent weights."""
        try:
            # Extract RL state dicts
            rl_states = []
            valid_weights = []
            
            for i, update in enumerate(model_updates):
                model_data = update.get('model_data', {})
                rl_checkpoint = model_data.get('rl_agent_checkpoint', {})
                
                if 'network_state_dict' in rl_checkpoint:
                    rl_states.append(rl_checkpoint['network_state_dict'])
                    valid_weights.append(weights[i])
            
            if not rl_states:
                logger.warning("No RL states found for aggregation")
                return None
            
            # Normalize weights
            total_weight = sum(valid_weights)
            if total_weight > 0:
                valid_weights = [w / total_weight for w in valid_weights]
            
            # Aggregate weights
            aggregated_state = {}
            param_names = rl_states[0].keys()
            
            for param_name in param_names:
                weighted_param = None
                
                for state_dict, weight in zip(rl_states, valid_weights):
                    if param_name in state_dict:
                        param_tensor = state_dict[param_name]
                        
                        if weighted_param is None:
                            weighted_param = weight * param_tensor
                        else:
                            weighted_param += weight * param_tensor
                
                if weighted_param is not None:
                    aggregated_state[param_name] = weighted_param
            
            logger.debug(f"Aggregated RL agent with {len(rl_states)} models")
            return aggregated_state
            
        except Exception as e:
            logger.error(f"RL aggregation failed: {e}")
            return None
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        if not self.aggregation_history:
            return {'total_aggregations': 0}
        
        successful_aggregations = sum(1 for h in self.aggregation_history if h['success'])
        total_models_aggregated = sum(h['num_models'] for h in self.aggregation_history)
        
        return {
            'total_aggregations': len(self.aggregation_history),
            'successful_aggregations': successful_aggregations,
            'success_rate': successful_aggregations / len(self.aggregation_history),
            'total_models_aggregated': total_models_aggregated,
            'avg_models_per_aggregation': total_models_aggregated / len(self.aggregation_history),
            'strategy': self.aggregation_strategy,
            'min_nodes_required': self.min_nodes_for_aggregation
        }


class AdaptiveAggregator(ModelAggregator):
    """
    Adaptive model aggregator that adjusts strategy based on performance.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history = []
        self.adaptation_threshold = 0.1
    
    async def aggregate_models(self, model_updates: List[Dict[str, Any]]) -> Optional[HybridNIDSModel]:
        """Aggregate models with adaptive strategy selection."""
        # Evaluate current strategy performance
        if len(self.performance_history) >= 5:
            recent_performance = np.mean(self.performance_history[-5:])
            
            # Switch strategy if performance is poor
            if recent_performance < self.adaptation_threshold:
                old_strategy = self.aggregation_strategy
                self.aggregation_strategy = 'weighted_average' if old_strategy == 'federated_averaging' else 'federated_averaging'
                logger.info(f"Adapted aggregation strategy from {old_strategy} to {self.aggregation_strategy}")
        
        # Perform aggregation
        result = await super().aggregate_models(model_updates)
        
        # Record performance (simplified metric)
        if result:
            performance_score = 1.0  # Success
        else:
            performance_score = 0.0  # Failure
        
        self.performance_history.append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
        
        return result
