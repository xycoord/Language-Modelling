import argparse
from typing import Dict, Any, Optional, List, Tuple

class ArgsParser(argparse.ArgumentParser):
    """ArgumentParser that automatically handles config loading with overrides"""
    
    def __init__(self, description: str = None, require_config: bool = True, **kwargs):
        super().__init__(description=description, **kwargs)
        
        # Add standard config arguments
        self.add_argument('--config', required=require_config, 
                         help='Path to config file')
        self.add_argument('--override', action='append', 
                         help='Override config values (key=value). Supports nested keys with dots: model_config.embed_dim=512')
        
        # Store for later use
        self._require_config = require_config
    
    def parse_config_args(self, args: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """Parse arguments and return config path and overrides"""
        parsed_args = self.parse_args(args)
        
        if not parsed_args.config and self._require_config:
            self.error("Config file is required")
        
        overrides = self._parse_overrides(parsed_args.override)
        
        return parsed_args.config, overrides
        
    
    def _parse_overrides(self, override_args: List[str]) -> Dict[str, Any]:
        """Parse override arguments with type inference"""
        if not override_args:
            return {}
        
        overrides = {}
        for override in override_args:
            if '=' not in override:
                self.error(f"Invalid override format: {override}. Expected key=value")
            
            key, value = override.split('=', 1)
            
            # Handle nested keys
            if '.' in key:
                main_key, sub_key = key.split('.', 1)
                if main_key not in overrides:
                    overrides[main_key] = {}
                overrides[main_key][sub_key] = self._infer_type(value)
            else:
                overrides[key] = self._infer_type(value)
        
        return overrides
    
    def _infer_type(self, value: str) -> Any:
        """Infer type from string value"""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # None/null
        if value.lower() in ('none', 'null'):
            return None
        
        # List (simple comma-separated)
        if ',' in value and not value.startswith('"'):
            return [self._infer_type(item.strip()) for item in value.split(',')]
        
        # Number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String (remove quotes if present)
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        
        return value