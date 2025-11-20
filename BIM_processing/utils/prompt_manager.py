"""
Prompt Manager for VLM queries
Loads and formats prompts from JSON configuration
"""
import json
import os
from typing import Dict, Any, Optional


class PromptManager:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize PromptManager with configuration file.
        
        Args:
            config_path: Path to JSON config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config directory
            # This file: /home/grass/.../BIM_processing/utils/prompt_manager.py
            # Config:    /home/grass/.../config/vlm_prompts.json
            current_dir = os.path.dirname(os.path.abspath(__file__))  # .../BIM_processing/utils
            parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to RoboMonitoring
            config_path = os.path.join(parent_dir, 'config', 'vlm_prompts.json')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Get formatted prompt by name.
        
        Args:
            prompt_name: Name of the prompt (e.g., 'door_detection_default')
            **kwargs: Variables to substitute in the prompt (e.g., robot_x, robot_y, threshold)
        
        Returns:
            Formatted prompt string
        """
        prompts = self.config.get('vlm_prompts', {})
        
        if prompt_name not in prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found in config")
        
        prompt_config = prompts[prompt_name]
        
        # Handle simple template prompts
        if 'template' in prompt_config:
            return prompt_config['template'].format(**kwargs)
        
        # Handle structured prompts (like door_detection_default)
        return self._build_structured_prompt(prompt_config, **kwargs)
    
    def _build_structured_prompt(self, prompt_config: Dict[str, Any], **kwargs) -> str:
        """Build a structured prompt from configuration."""
        parts = []
        
        # Add system context
        if 'system_context' in prompt_config:
            parts.append(prompt_config['system_context'])
        
        # Add task description
        if 'task' in prompt_config:
            parts.append(f"\n**Your Task:**")
            parts.append(prompt_config['task'])
        
        # Add current context with variables
        if kwargs:
            parts.append(f"\n**Current Context:**")
            if 'robot_x' in kwargs and 'robot_y' in kwargs:
                parts.append(f"- Robot position: ({kwargs['robot_x']:.3f}m, {kwargs['robot_y']:.3f}m) [x, y coordinates in meters]")
            if 'threshold' in kwargs:
                parts.append(f"- Search radius: {kwargs['threshold']}m")
        
        # Add steps
        if 'steps' in prompt_config:
            for step_config in prompt_config['steps']:
                parts.append(f"\n**Step {step_config['step']} - {step_config['name']}:**")
                for instruction in step_config['instructions']:
                    # Replace variables in instructions
                    formatted_instruction = instruction.format(**kwargs) if kwargs else instruction
                    parts.append(formatted_instruction)
        
        # Add answer format
        if 'answer_format' in prompt_config:
            parts.append(f"\n**Your Answer Format:**")
            for section in prompt_config['answer_format']['required_sections']:
                parts.append(f"{section['section']}: {section['instruction']}")
        
        # Add important notes
        if 'important_notes' in prompt_config:
            parts.append(f"\n**Important Notes:**")
            for note in prompt_config['important_notes']:
                parts.append(f"- {note}")
        
        parts.append("\nPlease analyze both sources of information and provide your assessment.")
        
        return '\n'.join(parts)
    
    def get_parsing_keywords(self, category: str) -> list:
        """
        Get keywords for parsing VLM responses.
        
        Args:
            category: 'positive_detection' or 'negative_detection'
        
        Returns:
            List of keywords
        """
        keywords = self.config.get('parsing_keywords', {})
        return keywords.get(category, [])
    
    def get_default_value(self, key: str) -> Any:
        """Get default configuration value."""
        defaults = self.config.get('default_values', {})
        return defaults.get(key)
    
    def list_available_prompts(self) -> list:
        """List all available prompt names."""
        return list(self.config.get('vlm_prompts', {}).keys())
    
    def reload_config(self):
        """Reload configuration from file (useful for hot-reloading)."""
        self.config = self._load_config()


# Example usage
if __name__ == '__main__':
    # Test the prompt manager
    pm = PromptManager()
    
    print("Available prompts:")
    for prompt_name in pm.list_available_prompts():
        print(f"  - {prompt_name}")
    
    print("\n" + "="*80)
    print("Example: door_detection_default")
    print("="*80)
    prompt = pm.get_prompt(
        'door_detection_default',
        robot_x=5.0,
        robot_y=3.0,
        threshold=2.0
    )
    print(prompt)
    
    print("\n" + "="*80)
    print("Example: door_detection_simple")
    print("="*80)
    prompt = pm.get_prompt(
        'door_detection_simple',
        robot_x=5.0,
        robot_y=3.0,
        threshold=2.0
    )
    print(prompt)
