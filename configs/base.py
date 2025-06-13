import os
from dataclasses import dataclass

@dataclass
class BaseEvalConfig:
    model_name = None 
    program = None
    cwd = None 

    def get_args(self):
        raise NotImplementedError("Subclasses must implement get_args()")

    def get_env(self):
        raise NotImplementedError("Subclasses must implement get_env()")

    def display_config(self):
        print(f"Model Name: {self.model_name}")
        print(f"Program Path: {self.program}")
        print(f"Working Directory: {self.cwd}")
        print("Arguments:", self.get_args())
        print("Environment Variables:", self.get_env())