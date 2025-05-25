from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import argparse
import torch
import copy
import os
import re
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

# def to_gif(images, path):

#     images[0].save(path, save_all=True,
#                    append_images=images[1:], loop=0, duration=len(images) * 20)

# def figure_to_image(figure):

#     figure.set_dpi(300)

#     figure.canvas.draw()

#     return Image.frombytes('RGB', figure.canvas.get_width_height(), figure.canvas.tostring_rgb())

# def image_grid(images, outpath=None, column_titles=None, row_titles=None):

#     n_rows = len(images)
#     n_cols = len(images[0])

#     fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
#                             figsize=(n_cols, n_rows), squeeze=False)

#     for row, _images in enumerate(images):

#         for column, image in enumerate(_images):
#             ax = axs[row][column]
#             ax.imshow(image)
#             if column_titles and row == 0:
#                 ax.set_title(textwrap.fill(
#                     column_titles[column], width=12), fontsize='x-small')
#             if row_titles and column == 0:
#                 ax.set_ylabel(row_titles[row], rotation=0, fontsize='x-small', labelpad=1.6 * len(row_titles[row]))
#             ax.set_xticks([])
#             ax.set_yticks([])

#     plt.subplots_adjust(wspace=0, hspace=0)

#     if outpath is not None:
#         plt.savefig(outpath, bbox_inches='tight', dpi=300)
#         plt.close()
#     else:
#         plt.tight_layout(pad=0)
#         image = figure_to_image(plt.gcf())
#         plt.close()
#         return image

# def get_module(module, module_name):

#     if isinstance(module_name, str):
#         module_name = module_name.split('.')

#     if len(module_name) == 0:
#         return module
#     else:
#         module = getattr(module, module_name[0])
#         return get_module(module, module_name[1:])

def set_module(module, module_name, new_module):

    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 1:
        return setattr(module, module_name[0], new_module)
    else:
        module = getattr(module, module_name[0])
        return set_module(module, module_name[1:], new_module)

def freeze(module):

    for parameter in module.parameters():

        parameter.requires_grad = False

def unfreeze(module):

    for parameter in module.parameters():

        parameter.requires_grad = True

# def get_concat_h(im1, im2):
#     dst = Image.new('RGB', (im1.width + im2.width, im1.height))
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (im1.width, 0))
#     return dst

# def get_concat_v(im1, im2):
#     dst = Image.new('RGB', (im1.width, im1.height + im2.height))
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (0, im1.height))
#     return dst

class StableDiffuser(torch.nn.Module):

    def __init__(self,
                scheduler='LMS'
        ):

        super().__init__()

        # load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae")
        
        # load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")
        
        # UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet")
        
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="feature_extractor")
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")

        if scheduler == 'LMS':
            self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        elif scheduler == 'DDIM':
            self.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        elif scheduler == 'DDPM':
            self.scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")    

        self.eval()

    def get_noise(self, batch_size, img_size, generator=None):

        param = list(self.parameters())[0]

        return torch.randn(
            (batch_size, self.unet.in_channels, img_size // 8, img_size // 8),
            generator=generator).type(param.dtype).to(param.device)

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(latents, noise, torch.tensor([self.scheduler.timesteps[step]]))

    def text_tokenize(self, prompts):

        return self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    def text_detokenize(self, tokens):

        return [self.tokenizer.decode(token) for token in tokens if token != self.tokenizer.vocab_size - 1]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def decode(self, latents):

        return self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample

    def encode(self, tensors):

        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.unet.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts, generator=None):

        noise = self.get_noise(n_imgs, img_size, generator=generator).repeat(n_prompts, 1, 1, 1)

        latents = noise * self.scheduler.init_noise_sigma

        return latents

    def get_text_embeddings(self, prompts, n_imgs):

        text_tokens = self.text_tokenize(prompts)

        text_embeddings = self.text_encode(text_tokens)

        unconditional_tokens = self.text_tokenize([""] * len(prompts))

        unconditional_embeddings = self.text_encode(unconditional_tokens)

        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings]).repeat_interleave(n_imgs, dim=0)

        return text_embeddings

    def predict_noise(self,
             iteration,
             latents,
             text_embeddings,
             guidance_scale=7.5
             ):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latents = torch.cat([latents] * 2)
        latents = self.scheduler.scale_model_input(
            latents, self.scheduler.timesteps[iteration])

        # predict the noise residual
        noise_prediction = self.unet(
            latents, self.scheduler.timesteps[iteration], encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * \
            (noise_prediction_text - noise_prediction_uncond)

        return noise_prediction

    @torch.no_grad()
    def diffusion(self,
                  latents,
                  text_embeddings,
                  end_iteration=1000,
                  start_iteration=0,
                  return_steps=False,
                  pred_x0=False,
                  trace_args=None,                  
                  show_progress=True,
                  **kwargs):

        latents_steps = []
        trace_steps = []

        trace = None

        for iteration in tqdm(range(start_iteration, end_iteration), disable=not show_progress):

            if trace_args:

                trace = TraceDict(self, **trace_args)

            noise_pred = self.predict_noise(
                iteration, 
                latents, 
                text_embeddings,
                **kwargs)

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(noise_pred, self.scheduler.timesteps[iteration], latents)

            if trace_args:

                trace.close()

                trace_steps.append(trace)

            latents = output.prev_sample

            if return_steps or iteration == end_iteration - 1:

                output = output.pred_original_sample if pred_x0 else latents

                if return_steps:
                    latents_steps.append(output.cpu())
                else:
                    latents_steps.append(output)

        return latents_steps, trace_steps

    @torch.no_grad()
    def __call__(self,
                 prompts,
                 img_size=512,
                 n_steps=50,
                 n_imgs=1,
                 end_iteration=None,
                 generator=None,
                 **kwargs
                 ):

        assert 0 <= n_steps <= 1000

        if not isinstance(prompts, list):

            prompts = [prompts]

        self.set_scheduler_timesteps(n_steps)

        latents = self.get_initial_latents(n_imgs, img_size, len(prompts), generator=generator)

        text_embeddings = self.get_text_embeddings(prompts,n_imgs=n_imgs)

        end_iteration = end_iteration or n_steps

        latents_steps, trace_steps = self.diffusion(
            latents,
            text_embeddings,
            end_iteration=end_iteration,
            **kwargs
        )

        latents_steps = [self.decode(latents.to(self.unet.device)) for latents in latents_steps]
        images_steps = [self.to_image(latents) for latents in latents_steps]

        # disable the safety checker
        # for i in range(len(images_steps)):
        #     self.safety_checker = self.safety_checker.float()
        #     safety_checker_input = self.feature_extractor(images_steps[i], return_tensors="pt").to(latents_steps[0].device)
        #     image, has_nsfw_concept = self.safety_checker(
        #         images=latents_steps[i].float().cpu().numpy(), clip_input=safety_checker_input.pixel_values.float()
        #     )

        #     images_steps[i][0] = self.to_image(torch.from_numpy(image))[0]

        images_steps = list(zip(*images_steps))

        if trace_steps:

            return images_steps, trace_steps

        return images_steps
   
class FineTunedModel(torch.nn.Module):

    def __init__(self,
                 model,
                 train_method,
                 lora_rank=None,
                 lora_alpha=1.0,
                 lora_init_prompt=None,
                 ):

        super().__init__()
        self.model = model
        self.ft_modules = {}
        self.orig_modules = {}
        self.lora_modules = {}
        self.module_forward_hooks = {}
        freeze(self.model)

        # collect lora module names
        lora_module_names = []
        for module_name, module in model.named_modules():
            if 'unet' not in module_name:
                continue
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if train_method == 'xattn':
                    if 'attn2' not in module_name:
                        continue
                elif train_method == 'xattn-strict':
                    if 'attn2' not in module_name or 'to_q' not in module_name or 'to_k' not in module_name:
                        continue
                elif train_method == 'noxattn':
                    if 'attn2' in module_name:
                        continue 
                elif train_method == 'selfattn':
                    if 'attn1' not in module_name:
                        continue
                elif train_method == 'full':
                    pass
                else:
                    raise NotImplementedError(
                        f"train_method: {train_method} is not implemented."
                    )
                lora_module_names.append(module_name)

        fisher_info_dict = None
        if lora_rank is not None and lora_init_prompt is not None:
            unfreeze(self.model)
            fisher_info_dict = self.compute_fisher_information(
                self.model, lora_init_prompt, lora_module_names
            )
            freeze(self.model)

        for module_name, module in model.named_modules():
            if 'unet' not in module_name:
                continue
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if train_method == 'xattn':
                    if 'attn2' not in module_name:
                        continue
                elif train_method == 'xattn-strict':
                    if 'attn2' not in module_name or 'to_q' not in module_name or 'to_k' not in module_name:
                        continue
                elif train_method == 'noxattn':
                    if 'attn2' in module_name:
                        continue 
                elif train_method == 'selfattn':
                    if 'attn1' not in module_name:
                        continue
                elif train_method == 'full':
                    pass
                else:
                    raise NotImplementedError(
                        f"train_method: {train_method} is not implemented."
                    )

                ft_module = copy.deepcopy(module)
                
                self.orig_modules[module_name] = module
                self.ft_modules[module_name] = ft_module
                
                if lora_rank is None:
                    unfreeze(ft_module)
                else:
                    freeze(ft_module)

                fisher_info = None
                if fisher_info_dict is not None:
                    fisher_info = fisher_info_dict.get(module_name, None)
                if lora_rank is not None:
                    lora_module = self._create_lora_module(module, lora_rank, lora_alpha, fisher_info)
                    self.lora_modules[module_name] = lora_module
                    unfreeze(lora_module)

        self.ft_modules_list = torch.nn.ModuleList(self.ft_modules.values())
        self.orig_modules_list = torch.nn.ModuleList(self.orig_modules.values())
        self.lora_modules_list = torch.nn.ModuleList(self.lora_modules.values())

    def _create_lora_module(self, module, rank, alpha, fisher_info=None):
        device = module.weight.device
        dtype = module.weight.dtype
        
        if isinstance(module, torch.nn.Linear):
            if fisher_info is not None:
                W = module.weight.data 
                F = fisher_info.detach()

                row_importance = F.sum(dim=1).sqrt().to(device=device)

                U, S, V = torch.svd_lowrank(row_importance[:,None] * W, q=rank)
                
                scaling_A, scaling_B = 1, 1

                lora_A = ((V * torch.sqrt(S)).t()) / scaling_A
                lora_B = (1/(row_importance+1e-5))[:,None] * (U * torch.sqrt(S)) / scaling_B

                W_star = (W - (lora_B@lora_A) / (scaling_A*scaling_B))

                lora_down = torch.nn.Linear(module.in_features, rank, bias=False).to(device=device, dtype=dtype)
                lora_up = torch.nn.Linear(rank, module.out_features, bias=False).to(device=device, dtype=dtype)
                
                lora_down.weight.data.copy_(lora_A)
                lora_up.weight.data.copy_(lora_B)

                module.weight.data.copy_(W_star)
            else:
                lora_down = torch.nn.Linear(module.in_features, rank, bias=False).to(device=device, dtype=dtype)
                lora_up = torch.nn.Linear(rank, module.out_features, bias=False).to(device=device, dtype=dtype)
                torch.nn.init.normal_(lora_down.weight, std=1.0/rank)
                torch.nn.init.zeros_(lora_up.weight)

            lora_down = lora_down.to(device)
            lora_up = lora_up.to(device)
            lora_module = LoRAModule(
                lora_down=lora_down,
                lora_up=lora_up,
                alpha=alpha,
                rank=rank
            )
        elif isinstance(module, torch.nn.Conv2d):
            if fisher_info is not None:
                W = module.weight.data
                F = fisher_info.detach()
                out_c, in_c, kh, kw = W.shape
                W2d = W.reshape(out_c, -1)
                F2d = F.reshape(out_c, -1)
                
                row_importance = F2d.sum(dim=1).sqrt().to(device=device)

                U, S, V = torch.svd_lowrank(row_importance[:,None] * W2d, q=rank)
                
                # align the rank of S 
                rank = min(rank, in_c * kh * kw, out_c)

                scaling_A, scaling_B = 10, 10

                lora_A = (V * torch.sqrt(S)).t() / scaling_A
                lora_B = (1/(row_importance+1e-5))[:,None] * (U * torch.sqrt(S)) / scaling_B

                W_star = W - ((lora_B@lora_A).reshape(out_c, in_c, kh, kw) / (scaling_A * scaling_B))
                
                lora_down = torch.nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=rank,
                    kernel_size=module.kernel_size,
                    padding=module.padding,
                    stride=module.stride,
                    bias=False,
                ).to(device=device, dtype=dtype)
                lora_up = torch.nn.Conv2d(
                    in_channels=rank,
                    out_channels=module.out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                ).to(device=device, dtype=dtype)

                lora_down.weight.data.copy_(lora_A.reshape(rank, module.in_channels, kh, kw))
                lora_up.weight.data.copy_(lora_B.reshape(module.out_channels, rank, 1, 1))

                module.weight.data.copy_(W_star)
            else:
                lora_down = torch.nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=rank,
                    kernel_size=module.kernel_size,
                    padding=module.padding,
                    stride=module.stride,
                    bias=False,
                ).to(device=device, dtype=dtype)
                lora_up = torch.nn.Conv2d(
                    in_channels=rank,
                    out_channels=module.out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                ).to(device=device, dtype=dtype)
                
                torch.nn.init.normal_(lora_down.weight, std=1.0/rank)
                torch.nn.init.zeros_(lora_up.weight)

            lora_down = lora_down.to(device)
            lora_up = lora_up.to(device)
            
            lora_module = LoRAModule(
                lora_down=lora_down,
                lora_up=lora_up,
                alpha=alpha,
                rank=rank
            )
        else:
            raise NotImplementedError(f"LoRA is not implemented for {type(module)}")
        return lora_module

    @staticmethod
    def compute_fisher_information(model, prompt, module_names):
        fisher_info = {name: torch.zeros_like(dict(model.named_modules())[name].weight) for name in module_names}
        diffuser = model
        diffuser.eval()
        criteria = torch.nn.MSELoss()
        iterations = 10
        nsteps = 50
        prompt = [prompt]
        
        for i in tqdm(range(iterations)):
            with torch.no_grad():
                index = np.random.choice(len(prompt), 1, replace=False)[0]
                erase_concept_sampled = prompt[index]
                
                
                neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
                positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
                target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
            

                diffuser.set_scheduler_timesteps(nsteps)

                iteration = torch.randint(1, nsteps - 1, (1,)).item()

                latents = diffuser.get_initial_latents(1, 512, 1)


                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                )

                diffuser.set_scheduler_timesteps(1000)

                iteration = int(iteration / nsteps * 1000)
                
                positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
                neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
                target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                if erase_concept_sampled[0] == erase_concept_sampled[1]:
                    target_latents = neutral_latents.clone().detach()
            
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)

            positive_latents.requires_grad = True
            neutral_latents.requires_grad = True
            
            loss = criteria(negative_latents, target_latents - (1*(positive_latents - neutral_latents)))
            
            loss.backward()
            for name in module_names:
                module = dict(model.named_modules())[name]
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    fisher_info[name] += module.weight.grad.data.pow(2)
        for name in fisher_info:
            fisher_info[name] /= iterations
        return fisher_info
    
    @classmethod
    def from_checkpoint(cls, model, checkpoint, train_method, lora_rank=None, lora_alpha=1.0, lora_init_prompt=None):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)

        ftm = FineTunedModel(model, train_method=train_method, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_init_prompt=lora_init_prompt)
        ftm.load_state_dict(checkpoint)

        return ftm
    
    def _hook_factory(self, module_name):
        def hook(module, input_tensor, output_tensor):
            # add LoRA results to outputs if LoRA module exists
            if module_name in self.lora_modules:
                lora_output = self.lora_modules[module_name](input_tensor[0])
                return output_tensor + lora_output
            return output_tensor
        return hook
        
    def __enter__(self):
        for key, ft_module in self.ft_modules.items():
            set_module(self.model, key, ft_module)

        for handle in self.module_forward_hooks.values():
            handle.remove()
        self.module_forward_hooks = {}
            
        # register the forward hook
        if self.lora_modules:
            for module_name, ft_module in self.ft_modules.items():
                if module_name in self.lora_modules:
                    hook = self._hook_factory(module_name)
                    handle = ft_module.register_forward_hook(hook)
                    self.module_forward_hooks[module_name] = handle

        return self

    def __exit__(self, exc_type, exc_value, tb):

        for key, module in self.orig_modules.items():
            set_module(self.model, key, module)
            
        for handle in self.module_forward_hooks.values():
            handle.remove()
        self.module_forward_hooks = {}

    def parameters(self):

        parameters = []

        for ft_module in self.ft_modules.values():

            parameters.extend(list(ft_module.parameters()))

        for lora_module in self.lora_modules.values():
            parameters.extend(list(lora_module.parameters()))

        return parameters

    def state_dict(self):

        state_dict = {key: module.state_dict() for key, module in self.ft_modules.items()}
        state_dict.update({f"lora_{key}": module.state_dict() for key, module in self.lora_modules.items()})
        return state_dict

    def load_state_dict(self, state_dict):

        for key, sd in state_dict.items():
            if key.startswith("lora_"):
                module_key = key[5:]
                if module_key in self.lora_modules:
                    self.lora_modules[module_key].load_state_dict(sd)
            elif key in self.ft_modules:
                self.ft_modules[key].load_state_dict(sd)


class LoRAModule(torch.nn.Module):
    def __init__(self, lora_down, lora_up, alpha, rank):
        super().__init__()
        self.lora_down = lora_down
        self.lora_up = lora_up
        self.scale = alpha / rank
        self.rank = rank
        
    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.scale