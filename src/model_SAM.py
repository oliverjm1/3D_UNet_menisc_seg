# my model, which is an alteration of the Segment Anything Model

import torch
import torch.nn as nn
from torch.nn import functional as F

class my_SAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.sigmoid = nn.Sigmoid()

        # freeze both the image encoder (for now) and prompt encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):

        # First get the image and prompt embeddings with no_grad
        # (only training decoder at the mo)
        with torch.no_grad():
            image_embedding = self.image_encoder(image)

            # no prompts used here, but getting empty embeddings so
            # that the mask decoder doesn't have to be altered
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                  points=None,
                  boxes=None,
                  masks=None,
            )

        # get low res masks using mask decoder
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False, # only want one mask
        )

        # postprocess masks to match original image/mask size
        input_size = (800, 1024)
        original_size = (200, 256)

        # upscale to input size
        masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # crop to remove padding
        masks = masks[..., : input_size[0], : input_size[1]]
        # resize to match original size of image
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        out = self.sigmoid(masks)
        return out