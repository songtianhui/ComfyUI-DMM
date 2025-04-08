import logging
import torch

import folder_paths
import comfy.utils
from comfy import model_detection, model_management, model_base
from comfy.sd import VAE, CLIP
from comfy.supported_models import SD15

from .dmm import DMMUNetModel


class DMM(SD15):
    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.BaseModel(self, model_type=self.model_type(state_dict, prefix), device=device, unet_model=DMMUNetModel)
        if self.inpaint_model():
            out.set_inpaint()
        return out


class DMMLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), )}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "dmm"

    def load_checkpoint(self, ckpt_name):
        embedding_directory=folder_paths.get_folder_paths("embeddings")
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        state_dict, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)

        # copy and modify from comfy.sd.load_state_dict_guess_config
        diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(state_dict)
        parameters = comfy.utils.calculate_parameters(state_dict, diffusion_model_prefix)
        weight_dtype = comfy.utils.weight_dtype(state_dict, diffusion_model_prefix)
        load_device = model_management.get_torch_device()

        unet_config = model_detection.detect_unet_config(state_dict, diffusion_model_prefix, metadata=metadata)
        model_config = DMM(unet_config)

        # unet
        unet_dtype = torch.float16
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(state_dict, diffusion_model_prefix, device=inital_load_device)
        model.load_model_weights(state_dict, diffusion_model_prefix)
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded diffusion model directly to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

        # vae
        vae_sd = comfy.utils.state_dict_prefix_replace(state_dict, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd, metadata=metadata)

        # clip
        clip_target = model_config.clip_target(state_dict=state_dict)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(state_dict)
            if len(clip_sd) > 0:
                parameters = comfy.utils.calculate_parameters(clip_sd)
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, model_options={})
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")


        left_over = state_dict.keys()
        if len(left_over) > 0:
            logging.debug("left over keys: {}".format(left_over))

        return (model_patcher, clip, vae, None)


class DMMApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "index": ("INT", ),}}
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "apply"

    CATEGORY = "dmm"

    def apply(self, model: model_base.BaseModel, index: int):
        # model.model.diffusion_model: DMMUNetModel
        model.model.diffusion_model.model_id = index
        return (model,)




NODE_CLASS_MAPPINGS = {
    "DMMLoader": DMMLoader,
    "DMMApply": DMMApply,
}