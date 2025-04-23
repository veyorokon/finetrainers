"""
Test module for the E2V dataset functionality.

This tests whether the IterableE2VDataset can correctly:
1. Identify elements from file suffixes
2. Apply the right conditioning types 
3. Preprocess elements according to configuration

Run test:
python tests/dataset_test_e2v.py -v
"""

import os
import sys
import json
import torch
import unittest
from pathlib import Path

# Add parent directory to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetrainers.trainer.e2v_trainer.data import IterableE2VDataset
from finetrainers.data import initialize_dataset


class MockDataset(torch.utils.data.IterableDataset):
    """Simple mock dataset for testing."""
    
    def __init__(self, files):
        self.files = files
        
    def __iter__(self):
        for file in self.files:
            video_file = file.get("video")
            caption_file = file.get("caption")
            image_files = file.get("images", [])
            
            yield {
                "video_path": video_file,
                "caption": open(caption_file, "r").read().strip() if caption_file else "",
                "images": image_files
            }


class TestE2VDataset(unittest.TestCase):
    """Test cases for E2V dataset functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Define test assets path
        self.assets_path = Path(__file__).parent.parent / "assets" / "tests" / "dataset"
        
        # Check if test assets exist
        if not self.assets_path.exists():
            raise FileNotFoundError(f"Test assets not found at {self.assets_path}")
            
        # Create configuration for testing
        self.config = {
            "data_root": str(self.assets_path),
            "dataset_type": "video_references",
            
            "elements": [
                {
                    "name": "object",
                    "suffixes": ["_object.png"],
                    "required": True,
                    "conditioning": "reference",
                    "vae": {"repeat": 4, "position": 0},
                    "clip": {"position": 0}
                },
                {
                    "name": "background",
                    "suffixes": ["_background.png"],
                    "required": False,
                    "conditioning": "reference",
                    "vae": {"repeat": 1, "position": 1}
                },
                {
                    "name": "character",
                    "suffixes": ["_mask.png"],
                    "required": False,
                    "conditioning": "reference",
                    "vae": {"repeat": 2, "position": 2}
                },
                {
                    "name": "captions",
                    "suffixes": [".txt"],
                    "required": True,
                    "conditioning": "text"
                },
                {
                    "name": "video",
                    "suffixes": [".mp4"],
                    "required": True,
                    "conditioning": "target"
                }
            ],
            
            "conditioning": {
                "reference": {
                    "type": "frame",
                    "frame_conditioning_type": "full",
                    "frame_conditioning_concatenate_mask": True,
                    "resolution": [256, 256]
                },
                "text": {
                    "type": "text",
                    "remove_common_llm_caption_prefixes": True
                },
                "clip": {
                    "type": "clip",
                    "resolution": [224, 224],
                    "preprocessor": "center_crop"
                },
                "target": {
                    "type": "video",
                    "resolution": [256, 256]
                }
            }
        }
        
        # Create mock dataset with test files
        self.test_id = "083"
        self.test_files = [
            {
                "video": str(self.assets_path / f"{self.test_id}.mp4"),
                "caption": str(self.assets_path / f"{self.test_id}.txt"),
                "images": [
                    str(self.assets_path / f"{self.test_id}_object.png"),
                    str(self.assets_path / f"{self.test_id}_background.png"),
                    str(self.assets_path / f"{self.test_id}_mask.png")
                ]
            }
        ]
        
        self.mock_dataset = MockDataset(self.test_files)
        
    def test_element_identification(self):
        """Test if dataset correctly identifies elements by suffix."""
        # Initialize dataset
        dataset = IterableE2VDataset(self.mock_dataset, self.config)
        
        # Get first item
        for item in dataset:
            processed = item.get("e2v_processed", {})
            
            # Check if all elements were identified
            self.assertIn("frame", processed, "Frame conditioning not found")
            self.assertIn("text", processed, "Text conditioning not found")
            
            # Check element identification
            frame_elements = processed.get("frame", {}).get("elements", {})
            self.assertIn("object", frame_elements, "Object element not identified")
            self.assertIn("background", frame_elements, "Background element not identified")
            self.assertIn("character", frame_elements, "Character element not identified")
            
            # We only need to check one item
            break
            
    def test_preprocessing(self):
        """Test if dataset correctly preprocesses elements."""
        # Initialize dataset
        dataset = IterableE2VDataset(self.mock_dataset, self.config)
        
        # Get first item
        for item in dataset:
            processed = item.get("e2v_processed", {})
            
            # Check if frame conditioning has the right structure
            self.assertIn("frame", processed)
            self.assertIn("elements", processed["frame"])
            self.assertIn("conditioning", processed["frame"])
            
            # Check frame conditioning parameters
            frame_conditioning = processed["frame"]["conditioning"]
            self.assertEqual(
                frame_conditioning.get("frame_conditioning_type"), 
                "full", 
                "Frame conditioning type mismatch"
            )
            self.assertTrue(
                frame_conditioning.get("frame_conditioning_concatenate_mask"),
                "Frame mask concatenation not enabled"
            )
            
            # Check if elements have tensors
            for element_name, element_data in processed["frame"]["elements"].items():
                self.assertIn("tensor", element_data, f"{element_name} missing tensor")
                self.assertIn("position", element_data, f"{element_name} missing position")
                
                # Check tensor shape
                tensor = element_data["tensor"]
                self.assertTrue(isinstance(tensor, torch.Tensor), f"{element_name} tensor not a Tensor")
                # Shape should be [B, C, T, H, W] (added frame dim)
                self.assertEqual(len(tensor.shape), 5, f"{element_name} tensor wrong dimension")
                
            # We only need to check one item
            break
            
    def test_position_ordering(self):
        """Test if elements respect position ordering from config."""
        # Initialize dataset
        dataset = IterableE2VDataset(self.mock_dataset, self.config)
        
        # Get first item
        for item in dataset:
            processed = item.get("e2v_processed", {})
            frame_elements = processed.get("frame", {}).get("elements", {})
            
            # Check positions match configuration
            if "object" in frame_elements:
                self.assertEqual(
                    frame_elements["object"].get("position"), 
                    0, 
                    "Object position incorrect"
                )
                
            if "background" in frame_elements:
                self.assertEqual(
                    frame_elements["background"].get("position"), 
                    1, 
                    "Background position incorrect"
                )
                
            if "character" in frame_elements:
                self.assertEqual(
                    frame_elements["character"].get("position"), 
                    2, 
                    "Character position incorrect"
                )
                
            # We only need to check one item
            break


if __name__ == "__main__":
    unittest.main()