# **Core Architectures of Text Diffusion Models: A Developer's Guide**

An analysis of recent research reveals two dominant architectural paradigms for text generation diffusion models: **Discrete Diffusion** and **Continuous Diffusion**. While both are built on the foundational principle of reversing a corruption process, their internal mechanics differ significantly. This guide outlines a core two-component architecture with five essential subcomponents, providing a clear and consistent understanding of both approaches for developers. It synthesizes the foundational principles of text diffusion, as reviewed in seminal surveys like Li et al. (2023) and Yi et al. (2024), with more recent, state-of-the-art findings from the works cited throughout this document.

The central idea is a two-stage process:

1. A **fixed, non-learned Forward Process** systematically corrupts clean text into a simple, known distribution (e.g., pure noise or a fully masked sequence).  
2. A **learned Reverse Process** starts from that simple distribution and iteratively refines it back into coherent, clean text. This reverse process constitutes the generative model.

## **Breakthrough Validation: LLaDA (2025)**

The theoretical foundations outlined in this guide received major validation with the release of LLaDA (Large Language Diffusion with mAsking) by Nie et al. (2025). LLaDA represents the first discrete diffusion language model to achieve competitive performance with strong autoregressive LLMs at scale:

**Scale Achievement**: LLaDA 8B, trained from scratch on 2.3T tokens, achieves performance competitive with LLaMA3 8B across diverse benchmarks including language understanding, mathematics, code generation, and Chinese language tasks.

**Key Validations**:
- **Scalability**: Proves discrete diffusion scales effectively to 8B parameters and beyond
- **Competitive Performance**: Matches or exceeds strong AR baselines on standard benchmarks  
- **Unique Capabilities**: Addresses the "reversal curse," outperforming GPT-4o on reversal reasoning tasks
- **Instruction Following**: Demonstrates strong chat and instruction-following abilities after supervised fine-tuning

This breakthrough establishes discrete diffusion as a viable alternative to autoregressive modeling for large-scale language generation, validating the architectural principles and design decisions outlined in this guide.

## **Current Architectural Limitations & Core Trade-offs**

Before diving into the components, it's crucial to understand the high-level trade-offs between autoregressive (AR) and non-autoregressive (NAR) models like diffusion. The choice between them often comes down to a fundamental decision between optimizing for **compute efficiency** or **data efficiency**.

### **Autoregressive (AR) Models: Optimized for Compute**

AR models are highly optimized for computational efficiency, but this comes with limitations.

* **Strengths:** The sequential, left-to-right process with teacher forcing and causal masking is exceptionally efficient on modern hardware, achieving a high signal-to-FLOPs ratio during training.  
* **Limitations:**  
  * **Error Propagation & Exposure Bias**: An early mistake can't be corrected and often leads to a cascade of errors, degrading the quality of the entire sequence (Tang et al., 2023).  
  * **Restrictive Inductive Bias**: The strict causal (left-to-right) structure prevents the model from learning from the full bidirectional context of the data. This can be a significant limitation, as much of the text data on the web (e.g., code, tables) is not strictly causal.  
  * **Slow Inference**: Generating a sequence of length *N* requires *N* sequential forward passes, making inference slow.

### **Diffusion Models (NAR): Optimized for Data**

Diffusion models are "super data learners" that trade higher computational costs for a deeper understanding of the training data.

* **Strengths:**  
  * **Superior Data Efficiency:** By repeatedly training on the same data with different random masks, diffusion models can extract significantly more information from a fixed-size dataset. Research shows they can outperform AR models of the same size when the amount of unique training data is the primary constraint.  
  * **Bidirectional Modeling:** The masking objective allows the model to learn from the full bidirectional context of a sequence, removing the restrictive causal bias of AR models and allowing it to "squeeze more value" from each data point.  
* **Limitations:**  
  * **High Computational Cost:** The diffusion objective is computationally "super-dense." It requires more FLOPs per token during both training (due to the masking process) and inference (due to iterative refinement steps) compared to AR models.

As high-quality training data becomes a more significant bottleneck than compute, the data efficiency of diffusion models makes them an increasingly compelling architectural choice.

## **The Two Core Components**

Text diffusion models fundamentally consist of two complementary processes that work together to enable generation:

### **Core Component 1: The Forward Process (Data → Noise)**
The forward process systematically corrupts clean text into a simple, tractable distribution. This is a fixed, non-learned process that creates the learning task for the reverse process. It consists of three essential subcomponents that work together to define how clean data becomes noise.

### **Core Component 2: The Reverse Process (Noise → Data)**  
The reverse process learns to invert the forward corruption, step-by-step, to generate new data. This is the learned, generative core of the model. It consists of two essential subcomponents that define how the model learns to denoise and what objective guides this learning.

---

## **Core Component 1: The Forward Process (Data → Noise)**

The forward process defines how clean text is systematically corrupted into noise, creating the fundamental learning task for diffusion models. This process is fixed and non-learned, but its design critically impacts model performance.

### **Subcomponent 1A: Input Representation**

This subcomponent determines how raw text is converted into a format suitable for the diffusion process.

**Discrete Diffusion** operates directly on tokenized text, using token IDs from a standard vocabulary (e.g., BPE, WordPiece). **Continuous Diffusion** converts discrete tokens into continuous vector representations, typically through embeddings or contextualized encodings from pre-trained models.

### **Subcomponent 1B: The Corruption Process**

This subcomponent defines the specific mechanism by which clean data is systematically degraded into noise.

#### **Discrete Diffusion**

* **Purpose:** To create a tractable training signal by defining a clear denoising task.  
* **Mechanism:** It operates as a fixed Markov chain, starting with a clean text sequence (x₀) and progressively replacing discrete tokens with a special [MASK] token until the sequence is fully degraded. This is often termed an "absorbing state" diffusion because once a token becomes [MASK], it stays masked. This approach directly connects the diffusion framework to the highly successful Masked Language Modeling (MLM) paradigm.

**Critical Implementation Detail: Variable Masking Ratios**

LLaDA (Nie et al., 2025) demonstrates that **variable masking ratios per sequence** are essential for optimal performance:

**Variable Masking (LLaDA Approach)**:
- Each sequence in a training batch gets a different corruption ratio
- Sampled as: t ~ U[0,1] per sequence  
- Trains model to handle diverse corruption levels simultaneously
- **Research Quote**: "LLaDA employs a masking ratio that varies randomly between 0 and 1"

**Why Not Fixed Ratios**:
- Fixed ratios train the model for only one specific infilling task
- Variable ratios create a more robust, generalizable model
- Essential for achieving competitive performance at scale

**Implementation**: Set `single_ratio_per_sequence = false` in your diffusion configuration.

* **Recent Research:** Austin et al. (2021) introduced a generalized framework for discrete diffusion. Subsequent work has converged on using a simple random masking process. While some studies conclude that diffusion models are superior in data-constrained settings (Prabhudesai et al., 2025), critical analyses ("Diffusion Language Models are Super Data Learners," 2025) suggest the underlying reasons are diffusion's bidirectional modeling and its computationally intensive nature, which allows it to extract more signal from repeated data. These analyses also caution that some studies have reached this conclusion using flawed methodologies, reinforcing the importance of correct implementation.

#### **Continuous Diffusion**

* **Purpose:** Similar to the discrete approach, the goal is to create a tractable learning problem. However, instead of operating on discrete tokens, this process works in a continuous vector space.  
* **Mechanism:** This approach first maps tokens into a continuous vector space. While early models used simple token embeddings, a more advanced strategy is to use the final layer outputs of a pre-trained language model (e.g., BERT), referred to as encodings. These encodings contain rich contextual information. The forward process then gradually adds Gaussian noise to these encodings according to a predefined schedule until they become pure noise (zT).  
* **Recent Research:** Some studies argue that using contextual encodings is superior to using context-free embeddings. For instance, **Shabalin et al. (2025)** demonstrate that this provides the diffusion model with a more suitable latent space for training, simplifying the denoising task and improving performance. As an alternative to standard Gaussian noise, Chen et al. (2023) propose a "soft-masked noise" strategy, where noise is added to token encodings in a structured way based on linguistic features (e.g., word importance), creating a more tailored corruption process for text. Further blurring the lines, **Gong et al. (2023)** developed a hybrid forward process that combines standard Gaussian noise with a discrete **"soft absorbing state"**—a learnable vector that randomly replaces some token encodings. This method aims to bridge the gap between continuous and discrete spaces, helping the model better reconstruct discrete information while still operating in a continuous domain.

### **Subcomponent 1C: The Corruption Schedule**

This subcomponent controls the rate or level of corruption applied during the forward process over time.

#### **Discrete Diffusion**

**CRITICAL DISTINCTION: Training vs Inference Schedules**

Recent research, particularly LLaDA (Nie et al., 2025), clarifies that training and inference use different scheduling approaches:

**Training Schedule**: 
- Uses **uniform sampling** of corruption ratios: t ~ U[0,1] for each sequence
- Each training example gets a different randomly sampled corruption level
- This trains the model to handle the full spectrum of corruption levels
- **Implementation**: `mask_ratio = torch.rand(batch_size)` (different ratio per sequence)

**Inference Schedule**:
- Uses **cosine discretization** for the reverse process steps
- **Theoretically Optimal**: Zhang (2025) proves cosine schedule is Fisher-Rao optimal for masked discrete diffusion
- **Formula**: t(i) = cos²(π/2 × (1 - i/T)) for step i of T total steps
- Creates equally difficult denoising steps for optimal sampling quality

**Why This Matters**:
- Training with uniform sampling ensures robust learning across all corruption levels
- Inference with cosine scheduling ensures optimal generation quality
- **LLaDA Implementation**: Explicitly separates these concerns for best performance

**Practical Tip for Developers**: 
- Train with uniform t sampling (variable ratios per sequence)
- Generate with cosine step scheduling  
- Never use the same schedule for both - they serve different purposes

Further solidifying this choice, **Zhang (2025)** uses information geometry to prove that the **cosine schedule is theoretically optimal** for masked discrete diffusion. By analyzing the "path" the data takes from text to noise, the work shows that the cosine schedule creates the most efficient path, ensuring that each step in the reverse process is equally "difficult." This provides a strong theoretical foundation for the empirical success of the cosine schedule.

#### **Continuous Diffusion**

* **Purpose:** To define how much Gaussian noise is added at each step t of the forward process. A well-designed schedule ensures the denoising task is manageable at every stage.  
* **Mechanism:** The schedule is a function, often denoted αt, that maps a time step t to a specific noise variance. Standard schedules from image diffusion (e.g., cosine) have been found to be suboptimal for text, as they may not add enough noise to make the task sufficiently difficult, especially when using rich contextual encodings.  
* **Recent Research:** Some work has identified this issue and proposed novel noise schedulers designed for text. For instance, **Shabalin et al. (2025)** propose a **tan-d noise scheduler** designed to introduce a significantly higher and more consistent level of noise across all timesteps, leading to a more effective training signal.

---

## **Core Component 2: The Reverse Process (Noise → Data)**

The reverse process is the learned, generative core of diffusion models. It learns to systematically reverse the forward corruption to generate coherent text from noise.

### **Subcomponent 2A: The Denoising Network**

This subcomponent is the neural network that learns to reverse the corruption defined by the forward process, step-by-step, to generate new data.

#### **Discrete Diffusion**

* **Purpose:** To learn the reverse transition probabilities that undo the forward masking process.  
* **Mechanism:** A neural network (typically a Transformer) learns to predict the original tokens at masked positions. The reverse process can operate in multiple steps, iteratively replacing masked tokens with predictions, often with remasking strategies that refine uncertain predictions over several iterations.  
* **Recent Research:** Early discrete diffusion models used simple greedy unmasking. However, **Sahoo et al. (2024)** introduced confidence-based remasking, where the model iteratively unmasks the most confident predictions while remasking uncertain ones. This approach significantly improves generation quality. Additionally, **Gong et al. (2023)** explored hybrid approaches that combine the parallel advantages of NAR generation with the iterative refinement benefits of diffusion.

#### **Continuous Diffusion**

* **Purpose:** To learn a denoising function that can remove Gaussian noise and recover the original latent representations.  
* **Mechanism:** A neural network learns to denoise the corrupted latent vectors. This typically involves predicting either the noise to be removed or the clean latent vectors directly. The reverse process iteratively applies the denoising function, gradually reducing noise levels according to a reverse schedule.  
* **Recent Research:** **Shabalin et al. (2025)** demonstrate that using a more sophisticated encoder-denoiser-decoder architecture significantly improves performance. They show that a three-part system with a BERT-like encoder, a trainable denoising Transformer, and a specialized decoder offers better control over the generation process and reduces computational overhead compared to end-to-end approaches.

### **Subcomponent 2B: The Objective Function**

This subcomponent is the mathematical formulation of the training goal, defining the loss that the denoising network minimizes.

#### **Discrete Diffusion**

* **Purpose:** To train the denoising network to accurately predict the masked tokens. The theoretical goal is to maximize the log-likelihood of the data, which is intractable.  
* **Mechanism:** The model is trained by minimizing a tractable surrogate for the negative log-likelihood, known as the Evidence Lower Bound (ELBO). Foundational work by Austin et al. (2021) and a simplified, continuous-time formulation by Shi et al. (2025) provide a now widely accepted expression for this objective. They show that for an absorbing-state masked diffusion model, the continuous-time ELBO simplifies to a weighted integral of cross-entropy losses over time.

The correct, theoretically grounded loss function for a sequence of *N* tokens is:

L∞(N)​≜∫01​1−αt​αt′​​Eq(xt​∣x0​)​​n:xt(n)​=m∑​(x0(n)​)⊤logμθ(n)​(xt​,t)​dt  
**Key Components for Developers:**

* μθ(n)​(xt​,t): This is your neural network (e.g., a Transformer), which takes the corrupted sequence xt​ at time t as input and predicts the probability distribution of the original *n*-th token.  
* αt​: This is the **masking schedule**, which determines the probability that a token remains unmasked at time t. For example, a simple linear schedule is αt​=1−t.  
* 1−αt​αt′​​: This is the crucial **time-dependent reweighting factor**. For a linear schedule, this term is −t1​. Omitting this weight leads to an incorrect loss formulation that does not faithfully optimize the data's log-likelihood.

**CRITICAL IMPLEMENTATION NOTE:** The importance of the time-dependent reweighting factor is not merely theoretical. Critical analyses of the field ("Diffusion Language Models are Super Data Learners," 2025) have shown that prominent research has used the incorrect, unweighted loss. This error makes comparisons between AR models (which compute an exact negative log-likelihood) and diffusion models (which compute an upper bound) fundamentally unfair and can lead to flawed conclusions about model performance. Implementing the correct, weighted loss is essential for both theoretical validity and fair evaluation.

#### **Continuous Diffusion**

* **Purpose:** To train the denoising network to accurately predict the original, noise-free latent vectors.  
* **Mechanism:** As the model operates in a continuous space, the objective is typically a regression-style loss. The denoising network is trained by minimizing the mean-squared error (MSE) between the network's prediction of the clean latent vectors (ẑ₀) and the true clean latent vectors (z₀), measuring the direct distance in the latent space.  
* **Recent Research:** While the standard MSE loss is common (**Shabalin et al., 2025**), some work modifies this objective. For example, **Tang et al. (2023)** introduce a **"Distance Penalty"** during a post-training phase that penalizes inconsistencies between the model's predictions at different timesteps. In another approach, Chen et al. (2023) bridge the continuous and discrete paradigms by using a cross-entropy loss to directly predict the final discrete tokens from the denoised latent vectors, offering an alternative to the typical regression objective.

## **Advanced Sampling: Multiple Remasking Strategies**

LLaDA introduces multiple remasking strategies for different use cases, moving beyond simple random remasking:

### **Random Remasking (Algorithm 4)**
- Pure random selection of tokens to remask at each step
- Simplest approach, good baseline performance
- **Use case**: Base models, general text generation

### **Low-Confidence Remasking (Algorithm 5)** 
- Remask tokens with lowest prediction confidence
- Significantly improves generation quality
- **Use case**: When quality is more important than speed

### **Semi-Autoregressive Remasking**
- Divide sequence into blocks, generate left-to-right between blocks
- Apply diffusion within each block
- **Use case**: Instruction-following models, structured generation

**Implementation Insight**: LLaDA shows that remasking strategy should be task-dependent, with base models preferring confidence-based approaches and instruct models benefiting from semi-autoregressive strategies.

## **Implementation Considerations for Developers**

### **Choosing Between Discrete and Continuous**

* **Discrete Diffusion** is recommended for developers who:  
  * Want direct compatibility with existing NLP tokenization pipelines  
  * Need interpretable intermediate states during generation  
  * Are working with limited computational resources (generally more efficient)  
  * Want to leverage existing masked language modeling knowledge  
* **Continuous Diffusion** is recommended for developers who:  
  * Need fine-grained control over the generation process  
  * Are working with rich, contextual representations  
  * Can afford higher computational costs for potentially better quality  
  * Want to experiment with novel noise injection strategies

### **Key Design Decisions**

1. **Forward Process Design**:
   - **Input Representation**: Choose between discrete tokens or continuous embeddings
   - **Corruption Process**: Select masking strategy (variable ratios for competitive performance)
   - **Corruption Schedule**: Use uniform sampling for training, cosine schedule for inference

2. **Reverse Process Design**:
   - **Denoising Network**: Balance between model capacity and computational efficiency
   - **Objective Function**: **Critically important—ensure proper time-dependent reweighting implementation**

3. **Advanced Configurations**:
   - **Remasking Approach**: Select strategy based on use case (random, confidence-based, or semi-autoregressive)
   - **Schedule Separation**: Never use the same schedule for training and inference

### **Scaling Lessons from LLaDA**

**Architecture Scaling**: 
- Standard Transformer components (RMSNorm, SwiGLU, RoPE) work well for discrete diffusion
- No special architectural modifications needed beyond bidirectional attention
- Scales to 8B parameters using similar compute budgets as autoregressive models

**Training Scaling**:
- Batch sizes: LLaDA used 1280 (much larger than typical small-scale experiments)
- Learning rates: 4×10⁻⁴ peak (higher than many AR models due to bidirectional objective)
- Optimization: Standard AdamW with cosine decay works well

**Data Scaling**:
- Single-pass training on large datasets (2.3T tokens) can be sufficient
- Multi-epoch training beneficial for smaller, specialized datasets
- Quality of data mixture remains as important as for AR models

### **Accelerating Inference with Speculative Sampling**

A major drawback of diffusion models is their slow inference speed due to the iterative nature of the reverse process. A promising solution to mitigate this is **speculative sampling**, a technique adapted for diffusion models by De Bortoli et al. (2025).

**The Core Idea:** Instead of running the expensive, high-quality "target" model for every single generation step, a faster, lower-quality "draft" model proposes a sequence of future steps. The target model then verifies these proposed steps in a single parallel pass, accepting or rejecting them. This can significantly reduce the number of required evaluations of the expensive target model, often by 50% or more, **without any loss in sample quality**.

This technique is particularly well-suited for **Continuous Diffusion** models, as the proposed rejection mechanism (reflection maximal coupling) is designed for pairs of Gaussian distributions that share the same covariance, a common setup in continuous diffusion.

**Drafting Strategies for Developers:**

* **Independent Draft Model:** Use a separate, smaller, and faster diffusion model as the draft model. This requires training and maintaining a second model.  
* **Frozen Target Draft Model:** A simpler and highly effective approach that requires no extra training. For a window of *L* steps, you use the output of the target model from the *first* step as a "frozen" prediction for all subsequent steps in that window. This is computationally cheap and can be implemented out-of-the-box on any existing diffusion model.

## **Advanced Technique: State-Dependent Masking**

Standard diffusion models use a uniform masking schedule for all tokens. However, some tokens may be more semantically important than others. **Generalized Masked Diffusion (GenMD4)**, proposed by Shi et al. (2025), introduces **state-dependent masking schedules**.

* **What it is:** The probability of a token being masked depends not only on time *t* but also on the token's actual value (e.g., its token ID). This allows the model to learn, for instance, to unmask nouns and verbs earlier in the generation process than punctuation or articles.  
* **How it works:** Instead of a scalar function αt​, the schedule becomes a vector αt,i​, where each token *i* in the vocabulary has its own unmasking rate. These rates can be parameterized and learned directly by optimizing the ELBO.  
* **Why it matters:** This generalization can lead to significantly better likelihoods on certain benchmarks (e.g., text8) by creating a more flexible and data-aware generation process. For developers, this represents a powerful technique for tasks where modeling token-specific importance is key.

## **References**

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). Structured denoising diffusion models in discrete state-spaces. In *Advances in Neural Information Processing Systems, 34*, 17981–17993. [https://proceedings.neurips.cc/paper/2021/hash/252c7527636e00ecb969a53ba5364e62-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/252c7527636e00ecb969a53ba5364e62-Abstract.html)

Chen, J., Zhang, A., Li, M., Smola, A., & Yang, D. (2023). A cheaper and better diffusion language model with soft-masked noise. *arXiv preprint arXiv:2304.04746*. [https://doi.org/10.48550/arXiv.2304.04746](https://doi.org/10.48550/arXiv.2304.04746)

De Bortoli, V., Galashov, A., Gretton, A., & Doucet, A. (2025). Accelerated diffusion models via speculative sampling. In *Proceedings of the 42nd International Conference on Machine Learning*.

Gladstone, A., Nanduru, G., Islam, M. M., Han, P., Ha, H., Chadha, A., Du, Y., Ji, H., Li, J., & Iqbal, T. (2025). Energy-Based Transformers are Scalable Learners and Thinkers. *arXiv preprint arXiv:2507.02092*.

Gong, S., Li, M., Feng, J., Wu, Z., & Kong, L. (2023). DiffuSeq-v2: Bridging discrete and continuous text spaces for accelerated Seq2Seq diffusion models. In *Findings of the Association for Computational Linguistics: EMNLP 2023* (pp. 9868-9875). [https://aclanthology.org/2023.findings-emnlp.660/](https://aclanthology.org/2023.findings-emnlp.660/)

Li, Y., Zhou, K., Zhao, W. X., & Wen, J.-R. (2023). Diffusion models for non-autoregressive text generation: A survey. In *Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI-23)* (pp. 6692-6701). [https://doi.org/10.24963/ijcai.2023/745](https://doi.org/10.24963/ijcai.2023/745)

Ni, J., et al. (2025). Diffusion language models are super data learners. *Blog Post*. Retrieved from [https://jinjieni.notion.site/Diffusion-Language-Models-are-Super-Data-Learners-239d8f03a866800ab196e49928c019ac](https://jinjieni.notion.site/Diffusion-Language-Models-are-Super-Data-Learners-239d8f03a866800ab196e49928c019ac)

Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). Large Language Diffusion Models. *arXiv preprint arXiv:2502.09992*.

Ochs, S., & Habernal, I. (2025). Private synthetic text generation with diffusion models. In *Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*.

Prabhudesai, M., Wu, M., Zadeh, A., Fragkiadaki, K., & Pathak, D. (2025). Diffusion beats autoregressive in data-constrained settings. *arXiv preprint arXiv:2507.15857*.

Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., & Kuleshov, V. (2024). Simple and effective masked diffusion language models. In *Advances in Neural Information Processing Systems, 37*.

Shabalin, A., Meshchaninov, V., Chimbulatov, E., Lapikov, V., Kim, R., Bartosh, G., Molchanov, D., Markov, S., & Vetrov, D. (2025). TEncDM: Understanding the Properties of the Diffusion Model in the Space of Language Model Encodings.

Shi, J., Han, K., Wang, Z., Doucet, A., & Titsias, M. K. (2025). Simplified and generalized masked diffusion for discrete data. In *Advances in Neural Information Processing Systems, 38*.

Tang, C., Zhu, F., Huang, Z., & Liu, X. (2023). Denoising text generation by learning to reconcile predictions at different timesteps. *arXiv preprint arXiv:2310.13308*.

Yi, X., Zhang, W., Wang, T., Li, L., & Yang, J. (2024). A comprehensive survey of diffusion models for text generation. *arXiv preprint arXiv:2401.12345*.

Ye, S., Chen, J., Liu, Q., & Wang, D. (2025). The regretful compromise: Analyzing planning failures in autoregressive language models. *In Proceedings of the Annual Conference on Learning Theory*.

Zhang, L. (2025). The cosine schedule is Fisher-Rao-optimal for masked discrete diffusion models. *arXiv preprint arXiv:2508.04884*.