from .dataset_manager import DatasetManager
class RandomPrompt:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "func"
    CATEGORY = "quicknodes"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "repository": ("STRING", {"default": "ChrisGoringe/flux_prompts"}),
                     "seed": ("INT", {"default" : 0})
                }}
    
    def func(self, repository, seed):
        self.dm = DatasetManager.instance(repository) # only loads if there is a change
        return (self.dm.get_random_prompt(seed), )
    