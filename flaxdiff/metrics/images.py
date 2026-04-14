from .common import EvaluationMetric
import jax
import jax.numpy as jnp
    
def get_clip_metric(
    modelname: str = "openai/clip-vit-large-patch14",
):
    from transformers import AutoProcessor, FlaxCLIPModel
    model = FlaxCLIPModel.from_pretrained(modelname, dtype=jnp.float16)
    processor = AutoProcessor.from_pretrained(modelname, use_fast=False, dtype=jnp.float16)
    
    @jax.jit
    def calc(pixel_values, input_ids, attention_mask):
        # Get the logits
        generated_out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
            
        gen_img_emb = generated_out.image_embeds
        txt_emb = generated_out.text_embeds

        # 1. Normalize embeddings (essential for cosine similarity/distance)
        gen_img_emb = gen_img_emb / (jnp.linalg.norm(gen_img_emb, axis=-1, keepdims=True) + 1e-6)
        txt_emb = txt_emb / (jnp.linalg.norm(txt_emb, axis=-1, keepdims=True) + 1e-6)

        # 2. Calculate cosine similarity
        # Using einsum for batch dot product: batch (b), embedding_dim (d) -> bd,bd->b
        # Calculate cosine similarity
        similarity = jnp.einsum('bd,bd->b', gen_img_emb, txt_emb)

        scaled_distance = (1.0 - similarity)
        # 4. Average over the batch
        mean_scaled_distance = jnp.mean(scaled_distance)

        return mean_scaled_distance
        
    def clip_metric(
        generated: jnp.ndarray,
        batch
    ):
        original_conditions = batch['text']
        
        # Convert samples from [-1, 1] to [0, 255] and uint8
        generated = (((generated + 1.0) / 2.0) * 255).astype(jnp.uint8)
        
        generated_inputs = processor(images=generated, return_tensors="jax", padding=True,)
        
        pixel_values = generated_inputs['pixel_values']
        input_ids = original_conditions['input_ids']
        attention_mask = original_conditions['attention_mask']
        
        return calc(pixel_values, input_ids, attention_mask)
    
    return EvaluationMetric(
        function=clip_metric,
        name='clip_similarity'
    )


def get_clip_score_metric(
    modelname: str = "openai/clip-vit-large-patch14",
):
    """Standard CLIPScore metric (Hessel et al. 2021).

    Returns 100 * max(cos(image_embed, text_embed), 0) averaged over the batch.

    HIGHER IS BETTER. Typical text-to-image models score in the 25-35 range on
    natural prompts. This is the metric used in the original CLIPScore paper and
    most diffusion model evaluations.

    Note: This is the CORRECT CLIP-based evaluation. The legacy `clip_similarity`
    metric in this file actually returns `1 - cos_sim` (a distance, lower is better)
    and should be considered deprecated. New experiments should use `clip_score`.
    """
    from transformers import AutoProcessor, FlaxCLIPModel
    model = FlaxCLIPModel.from_pretrained(modelname, dtype=jnp.float16)
    processor = AutoProcessor.from_pretrained(modelname, use_fast=False, dtype=jnp.float16)

    @jax.jit
    def calc(pixel_values, input_ids, attention_mask):
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        gen_img_emb = out.image_embeds
        txt_emb = out.text_embeds

        # Normalize embeddings
        gen_img_emb = gen_img_emb / (jnp.linalg.norm(gen_img_emb, axis=-1, keepdims=True) + 1e-6)
        txt_emb = txt_emb / (jnp.linalg.norm(txt_emb, axis=-1, keepdims=True) + 1e-6)

        # Cosine similarity per pair
        cos_sim = jnp.einsum('bd,bd->b', gen_img_emb, txt_emb)

        # CLIPScore: 100 * max(cos_sim, 0), averaged
        clip_score = 100.0 * jnp.maximum(cos_sim, 0.0)
        return jnp.mean(clip_score)

    def clip_score_metric(
        generated: jnp.ndarray,
        batch
    ):
        original_conditions = batch['text']
        generated = (((generated + 1.0) / 2.0) * 255).astype(jnp.uint8)
        generated_inputs = processor(images=generated, return_tensors="jax", padding=True,)
        pixel_values = generated_inputs['pixel_values']
        input_ids = original_conditions['input_ids']
        attention_mask = original_conditions['attention_mask']
        return calc(pixel_values, input_ids, attention_mask)

    return EvaluationMetric(
        function=clip_score_metric,
        name='clip_score',
        higher_is_better=True,
    )