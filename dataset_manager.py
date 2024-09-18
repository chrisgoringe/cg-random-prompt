
import requests, datasets
from tqdm import trange

class DatasetManager:
    filters = [
        lambda e : "score_9" not in e['prompt'],
        lambda e : "((" not in e['prompt'],
        lambda e : "<lora" not in e['prompt'],
    ]

    _instance = None
    @classmethod
    def instance(cls, dataset_id="ChrisGoringe/flux_prompts"):
        if cls._instance is None or cls._instance.dataset_id != dataset_id: 
            cls._instance = cls(dataset_id)
        return cls._instance
    
    def __init__(self, dataset_id:str, local=False):
        if local:
            self.ds = datasets.Dataset.load_from_disk(dataset_id)
            self.dataset_id = None
        else:
            self.ds = datasets.load_dataset(dataset_id)['train']
            self.dataset_id = dataset_id

    def update(self):
        new_prompts = self.prompts()
        for p in new_prompts: 
            self.ds = self.ds.add_item({"prompt":p})
        self.ds = datasets.Dataset.from_dict( {'prompt':self.ds.unique('prompt')} )
        self.clean()

    def clean(self):
        for f in self.filters:
            self.ds = self.ds.filter(f)

    def upload(self, dataset_id=None):
        dataset_id = dataset_id or self.dataset_id
        if dataset_id is None: raise Exception("Loaded locally and no dataset_id provided: can't upload")
        self.ds.push_to_hub(dataset_id)

    def prompts(self) -> list[str]:
        results = []
        for period in ["Week", "Day"]:
            for i in trange(1,6,desc=period):
                r = requests.get(url="https://civitai.com/api/v1/images", params={
                                    "limit":200,
                                    "modelVersionId":691639,
                                    "period":period,"page":i})
                for item in r.json().get('items',[]):
                    if (prompt:=(item.get('meta',None) or {}).get('prompt',None)): results.append(prompt.replace("\n"," "))
        return results
    
    def get_random_prompt(self, seed:int=None) -> str:
        self.shuffle(seed)
        return self.ds[0]['prompt']

    def shuffle(self, seed:int=None):
        self.ds = self.ds.shuffle(seed)

dataset_manager = DatasetManager.instance()

if __name__=='__main__': 
    dataset_manager.update()
    dataset_manager.upload()
    print(f"Dataset now contains {len(dataset_manager.ds)} prompts")