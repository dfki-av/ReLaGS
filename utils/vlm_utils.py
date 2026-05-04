import torch
import open_clip


class ClipSimMeasure:
    def __init__(self, clip_type="ViT-B-16"):
        if clip_type == "ViT-B-16":
            model, _, process = open_clip.create_model_and_transforms(
                "ViT-B-16",
                pretrained="laion2b_s34b_b88k",
                precision="fp16",
            )
            model.eval()
            self.clip_pretrained = model.to('cuda')
            self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        elif clip_type == "ViT-L-14-336":
            model, _, process = open_clip.create_model_and_transforms(
                "ViT-L-14-336",
                pretrained="openai",
                precision="fp16",
            )
            model.eval()
            self.clip_pretrained = model.to('cuda')
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
        else:
            raise NotImplementedError("Only ViT-B-16 and ViT-L-14-336 are supported.")
        
        # self.clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
        self.canon = ["object", "things", "stuff", "texture"]
        self.feature_dim = 512
        self.device = torch.device("cuda")
        self.loaded = False
    
    def load_model(self):
        # no need delayed loading
        self.loaded = True
        return

    def encode_text(self, text):
        text = self.tokenizer([text] + self.canon).to(self.device)
        with torch.no_grad():
            text_features = self.clip_pretrained.encode_text(text).type(torch.float32)
            text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).to(self.device)
        self.text_feature = text_features
        # return text_features
        
    def compute_similarity(self, semantic_feature):
        logit = semantic_feature @ self.text_feature.T
        positive_vals = logit[..., 0:1]  # rays x 1
        negative_vals = logit[..., 1:]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.canon))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2, should be argmin
        cos_sim = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.canon), 2))[:, 0, 0]
        return cos_sim

    def get_text_encoding(self, text_words):
        text = self.tokenizer(text_words).to(self.device)
        with torch.no_grad():
            text_features = self.clip_pretrained.encode_text(text).type(torch.float32)
            text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).to(self.device)
        return text_features


class DINOV3SimMeasure:
    def __init__(self):
        model, tokenizer = torch.hub.load("facebookresearch/dinov3", 'dinov3_vitl16_dinotxt_tet1280d20h24l', 
                                        weights="dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth", 
                                        backbone_weights="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
        model.eval()
        self.dinov3_pretrained = model.to('cuda')
        self.tokenizer = tokenizer
        self.canon = ["object", "things", "stuff", "texture"]
        self.feature_dim = 1024
        self.device = torch.device("cuda")
        self.loaded = False
    
    def load_model(self):
        # no need delayed loading
        self.loaded = True
        return
    
    def encode_text(self, text):
        text = self.tokenizer.tokenize([text] + self.canon).to(self.device)
        with torch.no_grad():
            text_features = self.dinov3_pretrained.encode_text(text)[:, 1024:]
            #text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).to(self.device)
            text_features = text_features.to(self.device)
        self.text_feature = text_features
        # return text_features
    
    def compute_similarity(self, semantic_feature):
        logit = semantic_feature @ self.text_feature.T
        positive_vals = logit[..., 0:1]  # rays x 1
        negative_vals = logit[..., 1:]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.canon))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10*sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2, should be argmin
        cos_sim = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.canon), 2))[:, 0, 0]
        return cos_sim
    
    def get_text_encoding(self, text_words):
        tokenized_texts_tensor = self.tokenizer.tokenize(text_words).to(self.device)
        with torch.no_grad():
            text_features = self.dinov3_pretrained.encode_text(tokenized_texts_tensor)[:, 1024:]
            #text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).to(self.device)
            text_features = text_features.to(self.device)
        return text_features
