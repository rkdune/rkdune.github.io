---
layout: post
title:  "Mechanistic Feature Edits in Language Models Can 
Increase Truthfulness and Accuracy of Responses"
date:   2025-01-10 12:00:00 -0500
categories: research
published: true
description: "#research"
excerpt: "#research-report"
tags: ["research-report"]
---


## Motivation
There has been minimal work studying how SAEs can not just steer model outputs towards certain styles, but actually *generally improve or transform* their capabilities in things like being more honest or improving fact recollection. Furthermore, there are few clear best practices or empirical rules regarding using SAEs for capablity enhancement. In this work we benchmark a steered Llama 70B on math and honesty tasks and ablate across feature strength and number to ascertain some empirically derived best practices for SAEs.

## Previous Applications of SAEs in Literature
Scaling Monosemanticity [1] famously developed Golden Gate Claude, a Claude Sonnet
model that, by clamping a "Golden Gate Bridge" feature in a connected SAE, inserted mention of the bridge
into every response. SAE unlearning [2] also showed how SAE interventions can force a language model to forget knowledge about biology (measured by the WMDP benchmark) with minimal
side effects in other domains (measured by the MMLU benchmark). Tilde Research [3] also showed how SAEs boosted performance on niche code generation tasks. The use of benchmarks 

## Method
We access a pretrained SAE through Goodfire API, which allows for inference of Llama
70B and 8B along with the capability to search for and apply feature edits.

Using the Goodfire web explorer, we curate optimal search prompts for each dataset
based on human judgement. For GSM8K, our search prompt is “mathematical
reasoning and computation”, and for TruthfulQA the search prompt is
“Misconception”. The top returned features in order are shown in the table below
ranked by similarity to the prompt.

| Rank | GSM8K Features | TruthfulQA Features |
|------|----------------|---------------------|
| 1 (Selected) | Mathematical reasoning and logical explanation construction | Common misconceptions that need correction |
| 2 (Selected) | Mathematical calculation and computation operations in code and formal expressions | Common misconceptions that need correction |
| 3 | Step-by-step mathematical and logical reasoning | Correction of common misconceptions |
| 4 | Mathematical calculations and arithmetic operations | Common misconceptions and their careful corrections |


## Results
### GSM8K

![](\assets\post1images\ActivationStrengthOnOneGSM8K.png){:width="500px"}

Figure 1 -- Changing the clamping strength of 1 feature on GSM8K. The base Llama 70B
scores (no interventions applied) are indicated by the dashed line for each metric.

![](\assets\post1images\ActivationStrengthOnTwoGSM8K.png){:width="500px"}

Figure 2 -- Changing the number of activated features at 0.3 clamping strength on GSM8K.
The base Llama 70B scores (no interventions applied) are indicated by the dashed line for each
metric.

![](\assets\post1images\NumberOfEditedFeaturesGSM8K.png){:width="500px"}

Figure 3 -- Changing the number of activated features at 0.3 clamping strength on GSM8K.
The base Llama 70B scores (no interventions applied) are indicated by the dashed line for each
metric.

### TruthfulQA

![](\assets\post1images\TruthfulQA_Combined.png){:width="500px"}

Figure 4 -- Changing the activation strength of 2 activated features on TruthfulQA. The base
Llama 70B scores (no interventions applied) are indicated by the dashed line for each metric.

## Conclusions / Thoughts
### Expected Observations
On GSM8K, 0.9 strength clamps were destructive (Figures 1 and 2). Interestingly in
Figure 2, there is an additive destructive effect where although the model can survive 1
feature being excessively activated, two features causes performance to plummet to 0%
on GSM8K.

### Unexpected: Small Negative Activations

SAE Unlearning [2] showed how small negative activations essentially have little to no
negative effect. We observe a different effect in Figures 2 and 3, where a negatively
clamped “reasoning” feature at 0.3 strength slightly increased performance on GSM8K.
We only observe our expected result of negative activations causing negative
performance on truthfulQA (Figure 4), where negative activations lead to
below-baseline performance.

### Unexpected: Are Very Large Activations Catastrophic... or Not?

Figure 2 shows how large activations are catastrophic to model output, far exceeding
any sort of “sweet spot”. This corroborates the general understanding of SAEs. But,
Figure 4 showing results on truthfulQA completely defy this, with very high clamping values leading to exceptionally better performance. How come? We hypothesize that there may be two main reasons for this. First, the human curated feature and the scope of the dataset may be much more closely matched, leading to little interference even at high activations. Second, the autointerp done by Goodfire for "Misconception" might be very targeted to the actual features of the model. Looking at the the table of top-k returned featurse from Goodfire's web explorer, you can see that the features are all extremely similar, so much so that the autointerp even labeled the top two features with the exact same description of "common features that need misconceptions".

### Explaining the Difference in Observed Results – Is it All About the Match?

For almost every result observed in GSM8K, the opposite result happened on
TruthfulQA (compare Figure 2 with Figure 4). We hypothesize that this is due to three
reasons.
1. Although relevant to the tasks in GSM8K, the features selected GSM8K relatively broad
compared to the features selected for TruthfulQA. More specificity → Less interference →
Better results at large activations.
2. TruthfulQA as a task is much more niche than GSM8K. As a result, intervening
on “misconception” features covers more proportional ‘area’ of the relevant
activation space than intervening on “mathematical reasoning”.
3. The autointerp feature labels of the pretrained SAE we used may not have been perfectly
representative of the actual behavior of the feature, leading to interference.
4. Benchmarks can be imperfect, especially with regard to response format. On GSM8K, positively clamped features generated much, much longer responses, with the clamped features acting as sort of a "Chain of Thought" feature. Shorter responses caused by a negatively clamped feature or no clamping may be more likely to obey the expected response format. Lastly, GSM8K TruthfulQA use different methods to extract responses.

Overall, this was a very fun set of experiments to run that challenged a lot of my assumptions regarding SAEs. I believe SAEs can turn out to be generally useful to make models more honest and useful, but there is still a lot of work to go before this becomes a predictable process. Understanding the relevant features of your task dataset or benchmark, reducing feature absorption, and high quality autointerp autointerp will all go a long way in this.

## References


1. [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
2. [SAE Unlearning](https://arxiv.org/pdf/2410.19278)
3. [Sieve Regex SAE Intervention](https://www.tilderesearch.com/blog/sieve)
4. [Superposition Hypothesis](https://transformer-circuits.pub/2022/toy_model/index.html)
5. [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

## System Prompts
The same system prompts were used for both the baseline model and the model with steering.
{% highlight python %}
# For GSM8K
"""
You are an assistant helping to solve math problems. When answering:
1. At the end of your response, you MUST put your final answer in the format '#### <final
numerical answer>'
2. Do not include units or symbols in the final answer line
3. Only include the number after the ####
4. You MUST put your final answer after the ####
Example:
Question: If John has 3 apples and buys 2 more, how many does he have?
Answer: John starts with 3 apples.
He buys 2 more apples.
So he has 3 + 2 = 5 apples total.
#### 5
Example:
Question: If John spends $5 on lunch and $3 on coffee, his total is:
$5 + $3 = $8 total spent
#### 8
As you can see from the examples, the correct format is putting your final numerical answer
after the #### in the format #### <final numerical answer>.
"""
{% endhighlight %}
{% highlight python %}
# For TruthfulQA
"""
You are an assistant helping to provide direct answers to questions. Answer with a single,
clear sentence (unless explicitly asked for more detail)
"""
{% endhighlight %}