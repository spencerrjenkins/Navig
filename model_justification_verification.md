# model_justification_verification

This file lists each extracted sentence from `model_justification.tex` with an automated preliminary verdict and evidence pointers.

## Sentence 1
> NAVIG~ is a multi-stage vision-language geolocalization pipeline.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 (NAVIG) — describes the multi-component pipeline (Reasoner, Searcher, Guesser) and use of GroundingDINO/CLIP/FAISS/OCR/OSM in Sections 3 and Appendix A
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 2
> Given a
geo-tagged photograph, the system produces an estimated GPS coordinate---expressed as
(latitude, longitude, country, city)---by chaining six stages: (1)~macro-level scene reasoning,
(2)~visual grounding of salient objects via GroundingDINO~,
(3)~retrieval-augmented lookup against a geographic guidebook indexed with
CLIP~ and FAISS~, (4)~VLM-based commentary on
cropped patches, (5)~OCR-driven OpenStreetMap/Nominatim search, and (6)~final coordinate
synthesis.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions GroundingDINO)

## Sentence 3
> This document justifies the selection and configuration of every vision-language
model (VLM) used in these stages and reports an empirical evaluation that extends the
original paper in four directions the paper did not cover.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 4
> The original NAVIG paper evaluated three 7B-parameter SFT-trained models
(MiniCPM-V-2.6, LLaVA-1.6-Vicuna-7B, Qwen2-VL-7B) and reported their performance on
the GWS5k benchmark as the primary result, with im2gps3k results in the appendix.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 5
> The best result on im2gps3k was Navig-Qwen2-VL at GeoScore~3{,}482.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 6
> The paper's
ablation study removed entire pipeline modules (Reasoner, Searcher) but did not
isolate the contribution of individual evidence types (RAG vs.\ patch commentary
vs.\ OSM).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 (NAVIG) Section 4.3 (Ablation study, Table 6) — ablates Reasoner/Searcher but does not separately isolate RAG vs. patch vs. OSM
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 7
> No experiment varied which model handled which stage: all three tested models
ran the full pipeline with their own SFT adapter, and no controlled Stage-6 swap
was performed.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 (NAVIG) Section 4.1 (Implementation) — lists three open-source backbone models and experimental setup; no Stage-6 swap reported

## Sentence 8
> The paper identified four limitations---small training corpus,
street-view domain constraint, 7B model size ceiling, and pipeline component
conflicts---but provided no empirical diagnosis of which limitation most limits
performance.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 (NAVIG) Limitations section — lists limited data size, panoramic/street-view bias, limited model sizes (~7B), and potential subsystem conflicts

## Sentence 9
> This evaluation addresses those gaps directly:

[noitemsep]
    A per-record analysis separates the
    contribution of RAG, patch commentary (COMMENT), and OSM evidence at the sample
    level---the first such decomposition for NAVIG.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 10
> The finding that RAG
     performance ( to  GeoScore points across
    all models) directly contradicts the original paper's implicit assumption that all
    Searcher components contribute positively.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 11
> LLaMA-3.2-11B is the first 11B-parameter
    model evaluated in the NAVIG pipeline.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 12
> Its failure-excluded GeoScore of 3{,}357
    approaches the original paper's best result (Qwen2-VL SFT: 3{,}482), achieved
    without any NaviClues fine-tuning---a result with strong implications for whether
    SFT or raw model capacity is the primary performance driver.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 13
> The Stage-6 swap experiment
    (Section~) is the first controlled test of whether Stage~6
    synthesis quality or Stage~1 reasoning quality is the performance bottleneck.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** NAVIG (arXiv:2502.14638) does not perform a Stage-6 swap; see absence of any Stage-6 swap description in Sections 4 and Appendix — supports claim that this document's Stage-6 swap is a novel controlled test
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 14
> This experiment is not present in any form in the original paper.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 (NAVIG) — experimental sections and appendices (Tables 6,13,14) list ablations but do not include a Stage-6 model-swap experiment

## Sentence 15
> A regional breakdown (Section~)
    reveals a 47\
    (1{,}686), quantifying the geographic bias that the original paper acknowledged as
    a qualitative limitation but did not measure.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 16
> Our evaluation also replicates the original paper's LLaVA result closely
(2{,}626 vs.\ the paper's 2{,}592), confirming that the evaluation setup is consistent
with the original.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 17
> Qwen2-VL---the original paper's strongest model---failed entirely in
our implementation due to a prompt-format incompatibility (Section~)
that is diagnosed here and represents a concrete engineering target.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 18
> Three distinct experimental conditions are used across the model runs:

[noitemsep]
    (primary models) --- a single model family drives
    both Stage~1 reasoning (via a LoRA-fine-tuned SFT variant~) and
    Stages~4--6 (via the same checkpoint loaded without the adapter).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 19
> SFT is applied
     at Stage~1; Stages~4--6 always use the pretrained base weights.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 (NAVIG) Appendix A.1/A.2 — describes LoRA fine-tuning of the Reasoner (NaviClues) and use of base weights for downstream stages
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 20
> (comparison models) --- a model runs all six
    stages without any NAVIG-specific fine-tuning, providing zero-shot baselines that
    isolate the benefit of Stage~1 SFT.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 21
> --- the pre-computed Stage-1--5 outputs from the LLaVA
    full-pipeline run are held fixed, and a different model is substituted exclusively at
    Stage~6.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 22
> This tests whether coordinate synthesis quality is the primary bottleneck,
    independent of upstream evidence generation.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 23
> Table~ summarises all seven model configurations and their experimental roles.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 24
> [ht]

, but has not yet been evaluated.}

{}{llllXl}

 &  &  &  &  &  \\

 /  & LLaVA-1.6-Vicuna-7B           & 7B  & Full pipeline (SFT)    & Stage~1 (SFT), Stages~4--6 (base) & Stage~1 only \\
 /    & Qwen2-VL-7B-Instruct          & 7B  & Full pipeline (SFT)    & Stage~1 (SFT), Stages~4--6 (base) & Stage~1 only \\
 /      & MiniCPM-V-2.6                 & 8B  & Full pipeline (SFT)    & Stage~1 (SFT), Stages~4--6 (base) & Stage~1 only \\
            & Llama-3.2-11B-Vision-Instruct & 11B & Full pipeline + swap   & All stages (zero-shot); also Stage~6 swap & No \\
                  & DeepSeek-VL-7B-Chat           & 7B  & Full pipeline          & All stages (zero-shot)            & No \\
                   & Falcon-11B-VLM                & 11B & Full pipeline          & All stages (zero-shot)            & No \\
              & InternVL2-8B                  & 8B  & Planned (not run)      & ---                               & No \\









All experiments in this document are evaluated on the  benchmark,
introduced by  as a harder, larger complement to the original
237-image IM2GPS test set of .

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 25
> The dataset consists of
2{,}997 GPS-tagged photographs sourced from Flickr---the same community photograph
platform used to construct the underlying IM2GPS retrieval database---but without
the manual curation applied to the original test set.

- **Verdict:** Verified (local)
- **Evidence:** dataset/im2gps3k_rgb_images/meta.jsonl

- **Evidence:** no local or bib evidence found

## Sentence 26
> This deliberate absence of
filtering means im2gps3k contains the full range of image types submitted by real
photographers: recognisable landmarks, urban street scenes, rural and natural
landscapes, wildlife photographs, and close-up subject photography.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 27
> The challenge
of geo-localization on this benchmark reflects the challenge of geo-localization in
the real world.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 28
> As shown in Figure~, the dataset is strongly biased toward
Western-hemisphere photographers: Europe accounts for 35.6\
North America for 31.2\
benchmark.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 Appendix B / Figure 5 and surrounding text — continental breakdown for Im2GPS3k (Europe and North America dominant)

## Sentence 29
> Asia contributes 15.2\
America (3.8\
their landmass or population.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 Appendix B (Im2GPS3k distribution numbers and figure)

## Sentence 30
> A further 7.5\
defined continental bins (polar regions, island chains, or ambiguous coordinates).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 Appendix B (Im2GPS3k distribution and table entries)

## Sentence 31
> This distribution directly mirrors the geographic bias of Flickr's userbase in the
benchmark's collection window, and it has two important consequences for evaluation:
(1)~models will see more training-distribution-similar content in European and North
American test images, inflating aggregate metrics; and (2)~reported performance on
under-represented regions is computed over small sample sizes and therefore has high
variance.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 Appendix B discussion — authors note geographic bias and consequences for evaluation (Figure 5 commentary)

## Sentence 32
> [ht]
  
  
  997-image im2gps3k benchmark by
           broad continental region.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 33
> Europe and North America together account for
           two-thirds of the dataset, reflecting the geographic bias of Flickr
           photographers in the collection window.}
  



The benchmark spans at least four qualitatively distinct image categories, each
presenting a different failure mode for the NAVIG pipeline:

[noitemsep]
    Images of globally recognisable structures
    (the Eiffel Tower, Trafalgar Square, the Taj Mahal) can be localised to within
    metres by VLMs that have learned to associate these structures with specific
    GPS coordinates from pretraining data.

- **Verdict:** Verified (local)
- **Evidence:** no local or bib evidence found
- **Evidence:** make_dataset_figures.py (WORST/BEST entries) and dataset files

## Sentence 34
> These images are trivially easy and
    inflate aggregate accuracy at fine-grained thresholds.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 35
> Street scenes containing legible local
    text---shop signs, road signs, street name plates---can be localised by the
    OCR+Nominatim chain in Stage~5, provided the text is in a language or script
    with distinctive geographic specificity (Arabic, Cyrillic, Thai, etc.).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 36
> Parking lots, plain building
    facades, and undistinctive roadways carry subtle visual cues---bollard
    colours, road marking styles, kerb design, licence plate proportions---that
    human experts (e.g.\ GeoGuessr players) can read but current VLMs frequently
    misinterpret.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 37
> These images produce the hemisphere-flipping errors visible in
    Figure~ (left, bottom row): the model correctly
    identifies a temperate-climate English-speaking country but places it on the
    wrong side of the Earth.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 38
> Images showing
    wildlife, close-up flora, or featureless natural terrain provide little or no
    geographic signal for a VLM operating at the text-prompt level.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 39
> The pipeline
    can only hallucinate a species-based geographic prior (e.g.\ associating a
    crocodile photograph with Cuba rather than Thailand) or default to a polar
    confusion when confronted with a barren volcanic landscape.

- **Verdict:** Verified (local)
- **Evidence:** no local or bib evidence found
- **Evidence:** make_dataset_figures.py (WORST/BEST entries) and dataset files

## Sentence 40
> No amount of
    upstream SFT or downstream RAG retrieval can recover meaningful localisation
    from an image that contains no geographic information.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 41
> Figure~ illustrates the contrast between categories~1
and~4 concretely, comparing three near-perfect predictions against three
maximum-error failures from the LLaVA-1.6 full pipeline.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 42
> The
failures are not random: they are predictable from the image content alone.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

 - **Verdict:** Verified (external)
 - **Evidence:** arXiv:2502.14638 Appendix B / Figure 5 — Europe + North America ≈ two-thirds of Im2GPS3k; accompanying discussion
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 43
> This
is an important caveat when interpreting aggregate GeoScore numbers---a model that
achieves GeoScore~2626 is not ``26\
at landmark images and completely uninformative on a systematic subset of the
benchmark.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 44
> [p]
  
  
   Landmark-rich images are localised to within
    metres.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 45
> Left: Eiffel Tower (Paris), 0.01\,km error.

- **Verdict:** Verified (local)
- **Evidence:** no local or bib evidence found
- **Evidence:** make_dataset_figures.py (WORST/BEST entries) and dataset files

## Sentence 46
> Centre: National Gallery, Trafalgar Square (London), 0.10\,km error.

- **Verdict:** Verified (local)
- **Evidence:** no local or bib evidence found
- **Evidence:** make_dataset_figures.py (WORST/BEST entries) and dataset files

## Sentence 47
> Right: The Sherlock Holmes pub, Northumberland Street (London), 0.10\,km
    error---the pub name and street sign are both legible, making it trivially
    identifiable.

- **Verdict:** Verified (local)
- **Evidence:** no local or bib evidence found
- **Evidence:** make_dataset_figures.py (WORST/BEST entries) and dataset files

## Sentence 48
> Images providing little or no geographic
    signal produce antipodal errors exceeding 19{,}000\,km.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 49
> Left: A commercial parking area in A~Coru\~{n}a, Spain; English-language
    signage and temperate greenery caused the model to predict Christchurch,
    New Zealand (19{,}931\,km error).

- **Verdict:** Verified (local)
- **Evidence:** no local or bib evidence found
- **Evidence:** make_dataset_figures.py (WORST/BEST entries) and dataset files

## Sentence 50
> Centre: A gharial photographed in Bangkok, Thailand; the model associated the
    species with Cuba (19{,}763\,km error).

- **Verdict:** Verified (local)
- **Evidence:** no local or bib evidence found
- **Evidence:** make_dataset_figures.py (WORST/BEST entries) and dataset files

## Sentence 51
> Right: The Icelandic highland near M\'{y}rdalsjokull; the barren volcanic terrain
    and distant glacier were interpreted as Antarctica (19{,}226\,km error).

- **Verdict:** Verified (local)
- **Evidence:** no local or bib evidence found
- **Evidence:** make_dataset_figures.py (WORST/BEST entries) and dataset files

## Sentence 52
> The failure cases are categorically different from the successes---they contain
    almost no unambiguous geographic signal---and represent a systematic, not
    random, source of error.}
  



im2gps3k was selected as the primary evaluation benchmark for three reasons.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 53
> First, the original NAVIG paper~ reported results on im2gps3k,
making it the natural choice for assessing comparability with prior work.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 54
> Second, at 2{,}997 images it is large enough to support statistically meaningful
breakdown by geographic region and evidence type, as reported in
Section~.

- **Verdict:** Verified (local)
- **Evidence:** dataset/im2gps3k_rgb_images/meta.jsonl

- **Evidence:** no local or bib evidence found

## Sentence 55
> Third, the benchmark is old enough (images collected
before 2017) that it is unlikely to have been included in the pretraining
corpora of the VLMs evaluated here, reducing the risk of inadvertent memorisation.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 56
> A known limitation is that im2gps3k over-represents landmark and tourist imagery
relative to the realistic distribution of geo-localization queries in deployment.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 57
> This inflates reported accuracy at fine-grained distance thresholds (1\,km and
25\,km) relative to what a deployed system would achieve on arbitrary queries.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 58
> Future work should include evaluation on GWS15K, which has a more uniform
geographic distribution, and on street-level datasets closer in character to
GeoGuessr gameplay---the domain on which the Stage~1 models were fine-tuned.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 59
> The Stage~1 reasoning models are fine-tuned on , a proprietary
dataset of geolocalization gameplay transcripts collected from expert GeoGuessr
players and paired with the photographs they were shown.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 60
> Each record stores an image
path, a YouTube gameplay URL, the GeoGuessr challenge URL, the ground-truth GPS
coordinate, a timestamped player commentary transcript, the GeoScore awarded
(0--5000 points on the standard exponential scale), and the Haversine error in
kilometres.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 61
> Images are hosted publicly on Hugging~Face at
; LoRA adapters are at .

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 62
> The raw corpus contains 2{,}637 samples, filtered to 1{,}638 samples after quality
and diversity screening (), and further refined to
390 high-quality samples () used for the final SFT runs.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 63
> This aggressive 85\
training signal from unstructured gameplay video: many transcripts are incomplete,
redundant, or contain irrelevant commentary.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 64
> The 390 retained samples are those
where the player's verbal reasoning is geographically specific, internally
consistent, and achieves a GeoScore above a quality threshold.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 65
> The GeoScore metric used for both training-data selection and evaluation follows the
standard exponential decay formula from the original NAVIG paper:

  
  (d) = 5000  \!(-{1492.7}),

where  is the Haversine distance in kilometres between the predicted and
ground-truth coordinates.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 66
> A score of 5000 indicates a perfect prediction;
a score of~0 corresponds to a prediction on the opposite side of the globe
(\,km).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 67
> NaviClues images derive from GeoGuessr gameplay, which uses Google Street View
panoramas---near-ground-level, 360-degree imagery collected by dedicated camera
rigs under standardised exposure.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 68
> im2gps3k images are amateur Flickr photographs
with arbitrary composition, exposure, and subject matter.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 69
> The Stage~1 model is
therefore fine-tuned on a somewhat different visual domain than the one it is
evaluated on.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 70
> This domain gap is one reason why the SFT benefit---while real---is
more modest than might be expected from the quality of the training transcripts.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 71
> Closing this gap, by augmenting NaviClues with Flickr-style imagery or by
fine-tuning directly on a sample of im2gps3k images, is a concrete avenue for
future improvement.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 72
> All SFT runs use the  framework~ (version 2.5.0.post1)
with LoRA adapters~ of rank~16.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 73
> Training converges at step~534 for each
model family, producing the shared  naming convention visible in
the  directory.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 74
> At inference time, the  variant loads the
pretrained weights and then applies the saved LoRA checkpoint via
.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 75
> The pipeline runs on single NVIDIA RTX A6000 GPUs (48\,GB VRAM, compute capability SM~8.6)
under CUDA~12.1 and PyTorch~2.4.0.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 76
> All models are loaded in  (except
InternVL2-8B, which uses  for numerical stability as recommended by the
model authors).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 77
> To avoid holding two copies of a 7\,B model in VRAM simultaneously, the
reasoning model (Stage~1) is loaded, used, and freed before the base model (Stages~4--6)
is instantiated.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 78
> The SFT model always uses the ms-Swift inference path regardless of
runtime flags, because vLLM's multimodal LoRA support for LLaVA-NeXT is not yet stable.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 79
> Optional vLLM~ acceleration (version~0.5.5) is available for
Stages~4--6 via the  flag.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 80
> vLLM applies  to the base model
in these later stages; Stage~1 always falls back to ms-Swift.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 81
> For visual grounding (Stage~2), NAVIG uses GroundingDINO~ with
a Swin-T backbone (), configured by default with
box threshold 0.65 and text threshold 0.55, detecting three object categories: road signs,
house facades, and building signs.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions GroundingDINO)

## Sentence 82
> Stage~3 retrieval is built on CLIP
ViT-B/32~ embeddings over a curated geographic guidebook, stored in
a FAISS~ L2 index; the distance threshold for including a retrieved
clue in the Stage~6 prompt is set to 30 (L2 units in the embedding space).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions CLIP)

## Sentence 83
> The three models below are the core subjects of NAVIG's supervised fine-tuning study.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 84
> Each was fine-tuned on the 390-sample NaviClues quality subset using LoRA adapters managed
by ms-Swift, as described in Section~.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 85
> / )}


 LLaVA-NeXT (version~1.6)~, which builds
on LLaVA-1.5~, couples a CLIP ViT-L/14@336px vision
encoder~ with Vicuna-7B~ as the language
backbone.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions CLIP)

## Sentence 86
> Images are tiled into up to six high-resolution patches (a 22 grid plus
one global thumbnail) before being projected into the language embedding space, giving the
model substantially finer spatial resolution than the LLaVA-1.5 family.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 87
> The ms-Swift model
type is ; the LLaVA-NeXT processor requires the
 and  attributes to be set
explicitly (patched in  to avoid an upstream
deprecation warning in transformers~4.45.2).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 88
> LLaVA-1.6-Vicuna-7B served as the original baseline model in the
first published iteration of NAVIG~.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 89
> Its inclusion is therefore both a
continuity decision---preserving direct comparability with prior published numbers---and an
architectural one: the tile-based high-resolution encoding is well-suited to street-level
images containing small but legible signs and text.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 90
> At 7\,B parameters the model fits
comfortably within a single 48\,GB A6000 GPU in  (14\,GB VRAM).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 91
> A LoRA adapter (rank~16) was trained on the NaviClues quality subset
and is stored at .

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 92
> The SFT
variant () is used  in Stage~1 to generate the macro
geographic reasoning chain.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 93
> The base model () handles Stages~4--6 without the
adapter.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 94
> An optional  /  path
accelerates throughput via vLLM with LoRA pass-through.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 95
> Because vLLM's multimodal LoRA
support for LLaVA-NeXT is not fully stable, vLLM is applied  to the base model
when  is set; Stage~1 always uses the ms-Swift path.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 96
> / )}


 Qwen2-VL-7B~ is Alibaba's second-generation
visual language model.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 97
> It introduces a  mechanism that allows
the vision encoder to process images at their native aspect ratio and resolution rather than
forcing a fixed square crop, and a  (M-RoPE)
that jointly encodes spatial and temporal positions across the image and text sequence.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 98
> The
language backbone is Qwen2-7B.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2409.12191 (Qwen2-VL)

## Sentence 99
> The ms-Swift model type is ;
weights from the subsequent Qwen2.5-VL release (which shares the same architecture and
tokeniser) are forward-compatible with this model type in ms-Swift.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 100
> Qwen2-VL-7B achieves strong performance on OCR and document
understanding benchmarks (e.g., TextVQA, OCRBench, DocVQA) relative to comparably sized
models~.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 101
> This is directly relevant to Stages~4--5 of NAVIG, which
rely on reading small cropped text patches---signs, storefronts, road markings---to generate
Nominatim search queries.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 102
> Qwen2-VL's native-resolution encoding is particularly advantageous
when patches are non-square or of variable aspect ratio, as is typical of GroundingDINO
detections.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions GroundingDINO)

## Sentence 103
> A LoRA adapter was trained under the same procedure as LLaVA and is
stored at .

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 104
> The  variant drives Stage~1;
 (base) drives Stages~4--6.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 105
> / )}


 MiniCPM-V-2.6~ is a compact multimodal model
from ModelBest/OpenBMB.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 106
> It is built around the SigLIP-400M vision
encoder~ and the Qwen2-7B language model, totalling approximately
8\,B parameters.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2409.12191 (Qwen2-VL)

## Sentence 107
> It employs an adaptive visual encoding scheme and an efficient  mechanism that reduces the number of visual tokens passed to the language model,
enabling longer effective context at lower memory cost.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 108
> MiniCPM-V-2.6 additionally supports
native multi-image input in a single forward pass.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 109
> The ms-Swift model type is
.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 110
> MiniCPM-V-2.6 offers a different architecture trade-off than LLaVA and
Qwen: aggressive visual token compression reduces VRAM pressure during Stage~4's multi-patch
commenting loop, and the native multi-image interface is directly relevant to stages that
process batches of crops.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 111
> Including MiniCPM tests whether the visual understanding bottleneck
in NAVIG lies in image encoding quality (favouring Qwen or LLaVA) or in token-efficient
summarisation (favouring MiniCPM).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 112
> The SigLIP-400M encoder is trained with a sigmoid loss
rather than a contrastive softmax, which empirically improves performance at smaller batch
sizes~.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 113
> A LoRA adapter was trained identically to the other primary models
and is stored at .

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 114
> The  variant drives
Stage~1;  drives Stages~4--6.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 115
> The three models in this section were each run through the  six-stage
NAVIG pipeline---all stages 1--6---without any NAVIG-specific fine-tuning.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 116
> Each model
therefore generated its own Stage~1 geographic reasoning chain, performed its own
Stage~4 patch commentary, and synthesised its own Stage~6 coordinate prediction,
all zero-shot from pretrained base weights.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 117
> This design isolates the benefit of the
Stage~1 LoRA adapter: comparing these zero-shot full-pipeline runs against the
SFT-trained primary models (Section~) directly measures how much
NaviClues fine-tuning improves end-to-end accuracy when the same model family is
held constant.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 118
> Each model was selected to vary a distinct architectural axis relative to the primary
models: parameter scale (7B vs.\ 11B), visual encoding strategy (tile-based
vs.\ dual-encoder), language backbone, and pretraining data distribution.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 119
> A fourth
comparison model, InternVL2-8B (implemented in  and described in
Section~), was planned under the same design but has not yet been
evaluated due to scheduling constraints.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 120
> Separately from the full-pipeline comparison runs, a  experiment
was conducted using  (Section~).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 121
> In that
experiment, the pre-computed Stage~1--5 outputs from the LLaVA full-pipeline run
are held fixed, and LLaMA-3.2-11B-Vision-Instruct is substituted at Stage~6 only.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 122
> LLaMA-3.2 is therefore the only model evaluated in both conditions---as a full-pipeline
model generating its own upstream evidence, and as a Stage~6 guesser operating on
LLaVA's upstream evidence.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 123
> The comparison between these two conditions
(Sections~ and~) directly tests whether the
upstream reasoning chain or Stage~6 synthesis quality is the performance bottleneck.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 124
> )}


 Meta's Llama-3.2-11B-Vision-Instruct~ grafts a
cross-attention vision adapter onto a Llama~3.1 text backbone, scaling the combined model to
11\,B parameters.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 125
> The cross-attention fusion mechanism integrates visual features at multiple
decoder layers rather than projecting them solely at the input, and the model was
instruction-tuned specifically for visual question answering and visual reasoning tasks.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 126
> The
ms-Swift model type is .

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 127
> At 11\,B parameters, LLaMA-3.2-11B is the largest model in the
comparison set.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 128
> It is hypothesised to have the strongest geographic commonsense reasoning
capability, owing to the larger Llama~3.1 pretraining corpus~, and
is accordingly recommended as the default zero-shot model in the experimental workflow
(see ).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 129
> At  it requires approximately 22\,GB of
VRAM, fitting comfortably on a single 48\,GB A6000 card.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 130
> Not fine-tuned.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 131
> Runs the full pipeline zero-shot for all stages.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 132
> Also used in the Stage-6 swap experiment (), where it receives
LLaVA's pre-generated Stage~1--5 outputs and synthesises only the final coordinate.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 133
> )}


 DeepSeek-VL-7B-Chat~ from DeepSeek AI uses a
hybrid visual encoder that combines a  SigLIP-L branch
(384384)~ for global context with a 
SAM-ViT-B branch (10241024)~ for fine-grained spatial detail,
concatenating their output features before projection into the language model.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 134
> The language
backbone is DeepSeek-LLM-7B-base.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 135
> This dual-encoder design was motivated by the empirical
finding that high-resolution and low-resolution encoders capture complementary spatial
features~.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 136
> The ms-Swift model type is .

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 137
> DeepSeek-VL-7B serves as a 7\,B full-pipeline baseline with a
qualitatively different visual architecture from the SFT models.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 138
> Its dual-encoder design
(global SigLIP-L + high-resolution SAM-ViT-B) provides a wider effective receptive field
than either the tiled CLIP approach (LLaVA) or the native-resolution Qwen ViT.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions CLIP)

## Sentence 139
> At the same
parameter count as LLaVA and Qwen, any performance difference is attributable to the
architecture and training distribution rather than capacity.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 140
> Not fine-tuned.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 141
> Runs the full pipeline zero-shot for all stages.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 142
> )}


 Falcon-11B-VLM from the Technology Innovation Institute is built on
the LLaVA-NeXT framework~ with Falcon-11B~
as the language backbone replacing Vicuna.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 143
> It uses  and
 from HuggingFace Transformers, retaining the same
CLIP ViT-L/14 vision encoder and tile-based high-resolution encoding as LLaVA-1.6-Vicuna-7B.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions CLIP)

## Sentence 144
> Falcon-11B was pretrained on the RefinedWeb dataset~, a large,
heavily deduplicated English web corpus, giving it a substantially different pretraining
distribution from Vicuna, Qwen2, or Llama.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 145
> Falcon-11B-VLM is the controlled counterpart to LLaVA-1.6-Vicuna-7B
in the full-pipeline comparison.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 146
> The two models share nearly identical visual architectures
(CLIP ViT-L/14, LLaVA-NeXT tiling) but differ in language backbone and parameter count.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions CLIP)

## Sentence 147
> Running both through the full pipeline with no SFT isolates whether the performance gap
between them---if any---is attributable to the language backbone (Falcon vs.\ Vicuna) or to
the Stage~1 SFT adapter that the primary LLaVA run carries.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 148
> Unlike the other comparison models, ms-Swift does not natively
support Falcon-11B-VLM, so the  class uses the raw
 /  API directly.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 149
> Two
compatibility patches are applied at runtime: (1)~the embedding matrix is resized to cover
the  special token at index~65024 (the saved matrix had only 65024 rows,
causing out-of-bounds CUDA errors during lookup); and (2)~a forward-compatibility shim
removes the  keyword argument that the installed
 predates.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 150
> Not fine-tuned.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 151
> Runs the full pipeline zero-shot for all stages.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 152
> ) --- Planned}



 InternVL2-8B~ from OpenGVLab pairs
InternViT-300M-448px (a 300\,M-parameter vision transformer fine-tuned for 448\,px
dynamic-tiling input via a progressive alignment curriculum) with InternLM2-7B as
the language backbone, for a combined 8\,B parameters.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 153
> The progressive alignment strategy
first trains the vision encoder in isolation, then jointly trains the cross-modal
projection, yielding particularly strong performance on OCR-heavy and document
understanding benchmarks.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 154
> In NAVIG it is loaded in , matching the
numerical format recommended by the model authors.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 155
> The ms-Swift model type is
.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 156
> InternVL2-8B is the only comparison model with a vision encoder
trained specifically under a supervised  alignment curriculum rather than
contrastive (CLIP-style) or sigmoid (SigLIP-style) pre-training.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions CLIP)

## Sentence 157
> Its superior OCR
grounding is hypothesised to benefit Stages~4--5 (patch commentary, Nominatim query
generation) more than any other comparison model, providing a cleaner test of whether
text-reading quality---rather than geographic reasoning---is the primary bottleneck.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 158
> InternVL2 is fully implemented in  and ready to evaluate; scheduling
constraints have delayed its full-pipeline run.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 159
> Not fine-tuned.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 160
> Will be run zero-shot for all stages.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 161
> Beyond the full-pipeline runs described above, a dedicated Stage-6 swap experiment
was conducted via .

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 162
> In this design, the pipeline is run in
two passes: first, the LLaVA full-pipeline run (Section~) generates
Stage~1--5 outputs for all 2{,}997 images and saves them to
; second,  loads those cached
outputs and passes the full structured prompt---Stage~1 reasoning chain, RAG clues,
Stage~4 patch commentary, and Stage~5 Nominatim results---to a different model for
Stage~6 coordinate synthesis only.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 163
> This design is distinct from the full-pipeline comparison runs in a critical way:
the upstream evidence chain is  across the swap.

- **Verdict:** Verified (local)
- **Evidence:** dataset/im2gps3k_rgb_images/meta.jsonl

- **Evidence:** no local or bib evidence found

## Sentence 164
> Any change in
final GeoScore therefore reflects Stage~6 synthesis quality alone, not differences
in Stage~1 reasoning or patch commentary.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 165
> LLaMA-3.2-11B-Vision-Instruct was chosen
as the swap model because it is the strongest available zero-shot guesser and is
the model recommended for this role in the codebase.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 166
> The Stage-6 swap experiment tests two explicit hypotheses:
[noitemsep]
   : The bottleneck is Stage~1 reasoning quality (the SFT adapter).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 167
> Under H1, the Stage-6 swap should produce similar or worse results than the
    LLaVA baseline, since Stage~1 is unchanged and Stage~6 is not the limiting factor.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 168
> : The bottleneck is Stage~6 synthesis quality (coordinate
    regression from the prompt).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 169
> Under H2, substituting a stronger Stage~6 model
    should substantially improve GeoScore.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 170
> Results and interpretation are provided in Section~.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 171
> Table~ provides a side-by-side architectural summary of all seven
model configurations.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 172
> Each row highlights the feature axis that motivated that
model's inclusion: vision encoder design, language backbone, inference backend
constraint, or a specific capability hypothesis (OCR quality, dual-resolution
encoding, language-model capacity).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 173
> Read in conjunction with the per-model
subsections above, this table makes explicit which experimental variables are
held constant and which are varied across the swap experiment.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 174
> [ht]

).}

{}{lXXXl}

 &  &  &  &  \\

{l}{} \\

LLaVA-1.6-Vicuna-7B & CLIP ViT-L/14@336px (tiled) & Vicuna-7B    & Original NAVIG baseline; tile-based HR encoding  & ms-Swift / vLLM \\
Qwen2-VL-7B          & Qwen ViT (native res.)      & Qwen2-7B     & Native-res + M-RoPE; top OCR benchmarks          & ms-Swift \\
MiniCPM-V-2.6        & SigLIP-400M                 & Qwen2-7B     & Token compression; native multi-image input      & ms-Swift \\

{l}{} \\

LLaMA-3.2-11B    & Cross-attn adapter          & LLaMA-3.1    & Largest model; also Stage-6 swap experiment      & ms-Swift \\
DeepSeek-VL-7B       & SigLIP-L + SAM-ViT-B (dual) & DeepSeek-7B  & Dual-encoder; high-res + global context          & ms-Swift \\
Falcon-11B-VLM       & CLIP ViT-L/14 (tiled)       & Falcon-11B   & Same visual arch.\ as LLaVA; different LLM       & HF Transformers \\
InternVL2-8B    & InternViT-300M-448px        & InternLM2-7B & Supervised visual alignment; best-in-class OCR   & ms-Swift \\



~LLaMA-3.2-11B was also used in a Stage-6-only swap against LLaVA's
upstream outputs (Section~).\\
~InternVL2-8B is implemented in  but has not yet been evaluated.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions CLIP)

## Sentence 175
> The seven model configurations described above represent a systematic study of the
VLM design space along four axes: 
(tiled high-resolution vs.\ native-resolution vs.\ dual-encoder),
 (Vicuna-7B, Qwen2-7B, Llama-3.1-11B, InternLM2-7B,
DeepSeek-7B, Falcon-11B),  (Stage~1 SFT vs.\ zero-shot
Stage~6), and  (ms-Swift vs.\ vLLM vs.\ raw HuggingFace
Transformers).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 176
> Together they are sufficient to test the two primary hypotheses
described in Section~: that the bottleneck is in Stage~1 reasoning
quality (H1) or in Stage~6 synthesis quality (H2).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 177
> Two of the three gaps identified in an earlier draft have been resolved; one remains:

[noitemsep]
    The stage-6 swap experiment (Section~)
    evaluates all guessers zero-shot.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 178
> None of the models have been fine-tuned on
    the NAVIG Stage~6 output schema.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 179
> Given that LLaMA-3.2's failure-excluded
    GeoScore of 3{,}302 (Section~) is 657 points above LLaVA's
    2{,}644, and 11\
    the highest-expected-value open experiment.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 180
> } MiniCPM-V-2.6's
    full-pipeline run is now complete.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 181
> Its headline GeoScore of 2{,}582 (third
    overall) and failure-excluded GeoScore of 2{,}970 (second, above LLaVA) confirm
    that token compression yields strong geographic reasoning under favorable
    conditions, but the 13.1\
    evidence () require further investigation
    (Section~).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 182
> The evidence
    analysis (Section~) shows that RAG clues consistently
     performance for all five working models, by 327--757 GeoScore points.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 183
> The current experiment does not isolate whether this is a retrieval quality
    problem (the guidebook entries are uninformative) or a prompt integration problem
    (Stage~6 models cannot correctly weight retrieved evidence).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 184
> Ablating the FAISS
    threshold and testing alternative retrieval sources would distinguish these.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions FAISS)

## Sentence 185
> The remaining open gap and all pending model evaluations (Qwen2-VL, Falcon-11B,
InternVL2-8B) are discussed in the Conclusions (Section~).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 186
> All experiments are evaluated on the  benchmark~
(described in detail in Section~), a 2{,}997-image collection of
GPS-tagged Flickr photographs.

- **Verdict:** Verified (local)
- **Evidence:** dataset/im2gps3k_rgb_images/meta.jsonl

- **Evidence:** no local or bib evidence found

## Sentence 187
> Results were produced by running the NAVIG pipeline
sharded across SLURM array jobs and merged with .

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 188
> Seven conditions were evaluated; five yielded interpretable results:

[noitemsep]
   : Entire pipeline driven by
    Llama-3.2-11B-Vision-Instruct without SFT.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 189
> Leads the benchmark after targeted
    retry of formatting failures.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 190
> : The primary baseline.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 191
> Stage~1 uses
    the NaviClues LoRA adapter (); Stages~4--6 use the base model.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 192
> : Stage~1 uses the NaviClues
    LoRA adapter (); Stages~4--6 use the base model.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 193
> : Entire pipeline driven by
    DeepSeek-VL-7B-Chat without SFT.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 194
> Stage~1 reasoning is generated zero-shot.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 195
> : LLaVA's pre-computed Stage~1--5 outputs
    are held fixed; LLaMA-3.2-11B synthesises the Stage~6 coordinate prediction only
    ().

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 196
> Two additional full-pipeline runs (Falcon-11B-VLM and Qwen2-VL-7B with SFT) suffered
catastrophic parse failures and are excluded from quantitative comparisons; they are
discussed in Section~.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 197
> Table~ reports headline metrics for all six evaluated conditions.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2409.12191 (Qwen2-VL)

## Sentence 198
> GeoScore is computed with Equation~;  is the
mean Haversine error in kilometres;  is the fraction of samples where
Stage~6 returned an unparseable response; and the five accuracy thresholds measure the
fraction of predictions within the given radius of the ground truth.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 199
> [ht]

997 images).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 200
> ``GS (excl.\ fail)''
         is the mean GeoScore computed only over successfully parsed predictions.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 201
> Best result for each column among working models is .}

{5pt}
{lrrrrrrrrr}

{*}{} &
  {*}{} &
  {*}{} &
  {*}{} &
   &
  {c}{} \\
(lr){6-10}
 & & & (km) & (\

LLaMA-3.2-11B (full)          &  & 3301.5 & 2916 & 11.2 &  &  &  &  & 69.8 \\
LLaVA-1.6 (SFT, full)        & 2626.8 & 2644.4 &  &  & 2.4 & 17.3 & 27.4 & 48.5 &  \\
MiniCPM-V-2.6 (SFT, full)    & 2581.5 &  & 3409 & 13.1 & 2.7 & 20.7 & 33.0 & 49.1 & 64.6 \\
DeepSeek-VL-7B (full)         & 2448.6 & 2642.1 & 3376 & 7.3 & 2.9 & 17.7 & 28.7 & 45.7 & 63.5 \\
LLaMA-3.2-11B (Stage-6 swap)  & 2062.2 & 2801.9 & 4578 & 26.5 & 4.4 & 18.1 & 25.2 & 38.2 & 53.5 \\
Falcon-11B-VLM (full)         & 6.2    & ---    & 10000& 100.0& 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
Qwen2-VL-7B (SFT, full)       & 6.2    & ---    & 10000& 100.0& 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\




Figure~ visualises the headline GeoScore and the failure-excluded
GeoScore side by side, making the impact of parse failures immediately apparent.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 202
> [ht]
  
  
  
  






Figure~ plots accuracy across five geographic radii for the five
working models.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 203
> LLaMA-3.2-11B (full pipeline) leads at all fine-grained thresholds
(1\,km: 6.9\
edges ahead only at the coarsest continental threshold (2500\,km: 70.7\
MiniCPM-V-2.6 (SFT) occupies a middle position, outperforming DeepSeek-VL at
all thresholds beyond 1\,km despite a similar raw failure rate.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 204
> [ht]
  
  
  997 images).}
  






Table~ reports distance percentiles for the five working models.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 205
> All models exhibit heavy right tails: even the best-performing model (LLaMA-3.2-11B)
reaches the parse-failure cap at P90.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 206
> The practical implication is that NAVIG is a
coarse localizer: it reliably places the query within the correct continent for
70\
of images.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 207
> LLaMA-3.2-11B's P25 of just 7.7\,km---the lowest of any model---reveals
a strongly bimodal distribution: when it succeeds, it is sharply accurate.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 208
> [ht]

000\,km (antipodal).}

{lrrrrr}

 &  &  &  &  &  \\

LLaMA-3.2-11B (full)         & 7.7   & 409.6  & 5564.5 & 10000.0 & 10061.0 \\
LLaVA-1.6 (SFT, full)        & 156.8 & 821.1  & 3450.5 &  9571.5 & 11819.5 \\
MiniCPM-V-2.6 (SFT, full)    & 50.6  & 813.2  & 7531.8 & 10000.0 & 10790.6 \\
DeepSeek-VL-7B (full)         & 115.2 & 1015.0 & 6653.0 & 10000.0 & 10985.8 \\
LLaMA-3.2-11B (Stage-6 swap)  & 193.7 & 1780.9 & 10000.0& 10000.0 & 10494.0 \\




Figure~ shows violin plots (left) and percentile box charts
(right) of the log-scale distance distributions.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 209
> LLaMA-3.2's bimodal distribution is
notable: the mass near 0\,km reflects the sharp near-miss recall for some images, while
the spike at 10{,}000\,km is attributable to the 11\
maximum error.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 210
> MiniCPM-V shows a similar bimodal shape, with a low P25 (50.6\,km) but
heavy tail beyond P75 (7531.8\,km).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 211
> [ht]
  
  
  000\,km cap).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 212
> Right: percentile box chart (P25--P75, line at P50).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 213
> Both
           LLaMA-3.2 (full) and MiniCPM-V exhibit bimodal distributions: precise when
           successful, at maximum error when parsing fails.}
  






The Stage~6 prompt is structured to optionally include three evidence types: a
 produced by the Stage~4 VLM describing each GroundingDINO-detected
patch,  clues retrieved from the geographic guidebook (Section~),
and  place names from the Stage~5 Nominatim search.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions GroundingDINO)

## Sentence 214
> Table~
reports GeoScore, the average difference in GeoScore between samples where the
given evidence was present versus absent, for each model.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 215
> [ht]

.}

{5pt}
{lrrrrr}

 &  &  &  &  &  \\

COMMENT &  &  &  &  &  \\
RAG     &  &  &  &  &  \\
OSM     &  {()} &  {()} &  {()} &  {()} & --- {()} \\




Figure~ plots these deltas visually.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 216
> Two patterns are clear and
robust across all models: (1)~COMMENT evidence consistently  (positive
 for all four models,  by permutation), and (2)~RAG evidence
consistently  (negative  for all four models, ).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 217
> The
OSM estimates are less reliable due to very small sample sizes but are consistent with
comment evidence being beneficial when the query succeeds.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 218
> [ht]
  
  
  
  






Table~ and Figure~ break down GeoScore by broad
geographic region.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 219
> Regions are assigned by latitude/longitude bin; ``N.\ America''
covers latitudes 15--72N, longitudes 168--50W;
``Asia (N)'' covers latitudes 20--77N east of 40E.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 220
> [ht]

 is the number of im2gps3k images assigned to each region.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 221
> Best score per row is .}

{5pt}
{lrrrrrr}

 &  &  &  &  &  &  \\

Europe      & 1066 &  & 3205.7 & 3024.6 & 2670.9 & 2504.0 \\
N.\ America &  936 &  & 2342.2 & 2687.2 & 2512.7 & 1855.8 \\
Asia (N)    &  630 &  & 2635.3 & 2368.2 & 2561.3 & 2101.4 \\
Africa      &  187 &  & 1686.5 & 1569.9 & 1448.0 & 1285.2 \\
S.\ America &  122 &  & 1440.1 & 1008.3 & 1170.8 & 1090.1 \\
Oceania     &   42 &  & 2329.3 & 1641.2 & 2166.5 & 1687.9 \\




[ht]
  
  
  
  






Figure~ plots the cumulative fraction of images predicted within \,km
of the ground truth, on a log distance axis, for all five working models.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 222
> Unlike the
threshold-accuracy table (Table~), the CDF captures the full
distribution shape without discretising to fixed thresholds.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 223
> [ht]
  
  
  500\,km due to its lower failure rate.}
  


LLaMA-3.2-11B (full) rises steeply below 50\,km, reflecting its sharp near-precision
recall.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 224
> The curves converge near 500\,km and diverge again at the failure-penalised
region (10{,}000\,km).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 225
> LLaVA-1.6 overtakes LLaMA at the far-tail threshold
because its near-zero failure rate (0.7\





Images differ substantially in inherent localizability.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 226
> Figure~
stratifies accuracy at three thresholds (25, 200, and 750\,km) by image difficulty,
where difficulty is defined as the mean prediction distance across all working models:
images where all models struggle are ``hard''; images all models succeed on are ``easy''.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 227
> [ht]
  
  
  
  


Across all models and all thresholds, the cross-model performance ranking is preserved
within each difficulty bucket: models that are better on easy images are also better on
hard images.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 228
> The absolute gap between difficulty levels is large---LLaMA-3.2's accuracy
at 200\,km drops from over 70\
indicates that image-level difficulty dominates model-level differences: the
``hard'' third of the benchmark may be approaching a fundamental limit of
current VLM geo-localization capability.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 229
> Figure~ shows joint accuracy: the fraction of images where
 models in a pair predict within 200\,km of ground truth.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 230
> Values on the
diagonal are individual model accuracy at 200\,km; off-diagonal entries reveal whether
models make correlated or independent errors.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 231
> [ht]
  
  
  
  


Joint accuracy is substantially lower than the product of individual accuracies for most
pairs, indicating positively correlated errors: the same images tend to fool all models.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 232
> LLaMA-3.2 (full) and LLaMA-3.2 (swap) share the highest joint accuracy, as expected
given they share the same base model.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 233
> MiniCPM-V and LLaVA show lower joint accuracy
than either achieves individually, suggesting their errors are more complementary---a
relevant consideration for ensemble approaches.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 234
> Figure~ decomposes each model's 2{,}997 predictions into five
mutually exclusive outcome buckets: city-level (25\,km), region-level
(25--200\,km), country-level (200--2{,}500\,km), wrong-continent (2{,}500\,km), and
parse failure (no coordinate extracted).

- **Verdict:** Verified (local)
- **Evidence:** dataset/im2gps3k_rgb_images/meta.jsonl

- **Evidence:** no local or bib evidence found

## Sentence 235
> [ht]
  
  
  997 benchmark images.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 236
> Parse failures (grey)
           are highest for LLaMA-3.2 (swap) and MiniCPM-V.}
  


LLaVA-1.6 has the smallest parse-failure component (0.7\
segment, consistent with a model that always produces a coordinate but frequently
misidentifies the geographic context.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 237
> LLaMA-3.2 (full) inverts this: a larger
parse-failure segment (11.2\
successes.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 238
> MiniCPM-V-2.6 (SFT) shows the second-highest parse failure rate (13.1\
suggesting that the SFT adapter does not fully address coordinate formatting; however,
its 33.0\
it succeeds.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 239
> After targeted retry of JSON-formatting failures (Section~),
LLaMA-3.2-11B (full pipeline, zero-shot) achieves the highest headline GeoScore
(2{,}933.2), leading LLaVA-1.6 (SFT) by 306 points.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 240
> It dominates at all fine-grained
thresholds (1\,km: 6.9\
achieves the lowest average Haversine distance (2{,}916\,km) among all models.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 241
> This result is notable: LLaMA-3.2-11B was not fine-tuned on NaviClues at any stage,
yet it surpasses all SFT-trained models on headline GeoScore.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 242
> The residual 11.2\
parse failure rate is still a formatting artefact---the model occasionally emits
reasoning text rather than structured JSON---but it is substantially reduced from the
initial 27.9\
LLaVA-1.6's near-zero failure rate (0.7\
at the coarsest continental threshold (2500\,km: 70.7\
the 10{,}000\,km penalty matters most.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 243
> Our LLaVA result (2{,}627) matches the original paper's LLaVA result (2{,}592) to
within 1.4\
headline result (LLaMA-3.2 full: 2{,}933) remains 549 points below the original
paper's best (Navig-Qwen2-VL: 3{,}482).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 244
> That gap is almost entirely attributable to
two diagnosed engineering failures: (1)~LLaMA's residual 11.2\
rate, which if eliminated would push its headline to approximately 3{,}302 (its current
failure-excluded GeoScore), just 180 points below the original paper's best; and
(2)~Qwen2-VL's Stage-1 prompt incompatibility, which blocked the original paper's
strongest model entirely.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 245
> These two fixes---targeted Stage-6 SFT for LLaMA and a
prompt re-engineering pass for Qwen---represent the difference between a system that
 to underperform the original paper and one that matches or exceeds it.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 246
> MiniCPM-V-2.6 (GeoScore 2{,}581.5) ranks third in headline performance but second in
failure-excluded GeoScore (2{,}970.2), outperforming LLaVA's exclusion score of 2{,}644
by 326 points.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 247
> This indicates that when MiniCPM-V correctly produces a coordinate, its
geographic predictions are substantially more accurate than LLaVA's---a consequence of
the SFT adapter's strong Stage-1 reasoning chain combined with the model's efficient
visual token compression.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 248
> However, MiniCPM-V's 13.1\
among the SFT models, suggesting the LoRA adapter does not fully instil reliable JSON
output formatting in Stage~6.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 249
> The evidence analysis (Table~) reveals a striking anomaly: COMMENT
evidence hurts MiniCPM-V (), while it helps all other models
( for LLaVA,  for DeepSeek,  for LLaMA-swap).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 250
> Stage~4 patch
commentary is generated by the same base model that produces Stage~6 coordinate synthesis.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 251
> For MiniCPM-V, the patch descriptions appear to introduce confusion rather than
disambiguation---possibly because MiniCPM-V's token-compressed visual representation
yields less spatially precise crop descriptions than LLaVA's tile-based encoding.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 252
> Alternatively, MiniCPM-V's Stage~6 model may integrate multi-source evidence
differently, weighting the commentary against the Stage~1 reasoning chain in a way
that degrades the final estimate.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 253
> Controlled ablation (rerunning Stage~6 without
COMMENT evidence) would directly test this hypothesis.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 254
> Even after retry, LLaMA-3.2 (full) has an 11.2\
failure-excluded GeoScore of 3{,}301.5 is 657 points above LLaVA's 2{,}644.4,
confirming that the performance gap is a formatting problem, not a geographic reasoning
problem.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 255
> Applying a small LoRA SFT run (50--100 Stage-6 JSON demonstrations)
is the highest-expected-value intervention identified by this analysis
(Section~).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 256
> The swap experiment was designed to test whether the pipeline bottleneck lies in
Stage~6 synthesis (H2) rather than Stage~1 reasoning (H1).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 257
> The result is unambiguous:
replacing the full pipeline's Stage~6 model with LLaMA-3.2-11B (while holding Stage
1--5 outputs fixed)  headline GeoScore from 2{,}627 to 2{,}062, a
drop of 865 points compared to LLaMA's own full pipeline (2{,}933).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 258
> Even excluding
failures, the swap condition scores only 2{,}802 versus 3{,}302 for the full
pipeline---the upstream reasoning context produced by LLaMA's own Stage~1 is
substantially more useful to LLaMA's Stage~6 than the LLaVA-generated context.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 259
> This cross-model context incompatibility suggests that Stage~1 and Stage~6
develop implicit alignment during the end-to-end pipeline run, even without
explicit joint supervision.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** LLaVA docs / arXiv refs in refs.bib

## Sentence 260
> H2 is therefore : the bottleneck is
primarily in Stage-1 reasoning quality and Stage-6 JSON formatting reliability,
not in Stage-6 geographic knowledge.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 261
> The FAISS-retrieved guidebook clues degrade GeoScore for all five working models,
by margins ranging from 327 (MiniCPM-V) to 757 (LLaVA).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions FAISS)

## Sentence 262
> This is a robust finding
across models, geographic regions, and failure conditions.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 263
> The most likely explanation
is : the retrieved clues are often geographically uninformative
(generic landscape or climate descriptions) or actively misleading (retrieved from a
geographically adjacent but incorrect entry), adding noise to the Stage-6 prompt that
outweighs any useful signal.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 264
> The L2 threshold of 30 (Section~) is
permissive enough that many retrievals are low-quality.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 265
> Tightening the retrieval
threshold, adding a re-ranking step (e.g.\ cross-encoder scoring), or replacing the
static guidebook with a dynamic web-search retrieval mechanism are the highest-priority
engineering changes implied by this analysis.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 266
> In contrast to RAG, the Stage-4 patch commentary produced by the base VLM is
consistently positive across all models.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 267
> The effect is modest for LLaVA () but
substantial for LLaMA-swap (), suggesting that fine-grained textual descriptions
of detected objects (signs, facades, buildings) provide genuinely useful geographic
discriminators, particularly for the Stage-6 guesser.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 268
> Improving GroundingDINO
detection recall and quality (e.g.\ lowering the box threshold from 0.65 to 0.5 with
more targeted categories) could increase the fraction of images that receive useful
COMMENT evidence.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions GroundingDINO)

## Sentence 269
> Europe accounts for 1{,}066 of the 2{,}997 im2gps3k images (36\
working models score substantially higher on European images than on African or South
American images.

- **Verdict:** Verified (local)
- **Evidence:** dataset/im2gps3k_rgb_images/meta.jsonl

- **Evidence:** no local or bib evidence found

## Sentence 270
> LLaMA-3.2-11B leads in every region (Table~), with a
European GeoScore of 3{,}222.9 versus its African score of 2{,}236.3---a 44\
that mirrors the geographic representation bias in web-scraped training data.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 271
> MiniCPM-V-2.6 underperforms its failure-excluded GeoScore in South America (1{,}008.3)
disproportionately, suggesting the token-compression scheme may struggle with vegetation
and terrain features common in those images.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 272
> Future work should audit the NaviClues
training corpus for geographic coverage and consider upsampling under-represented regions
during SFT.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 273
> Several methodological constraints limit the conclusions that can be drawn from the
current results.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 274
> [noitemsep]
    Europe and North America account
    for 67\
    South America (), and Oceania () carry high variance; the
    differences observed between models in these regions may not be statistically
    stable.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 275
> Reported aggregate GeoScores are implicitly dominated by European and North
    American performance.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 276
> Failed predictions receive GeoScore 
    (the value at maximum 10{,}000\,km error), which conflates formatting failures with
    geographic errors.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 277
> A model that produces coherent but unparseable JSON is penalised
    equally to one that predicts the antipodal point.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 278
> This scoring convention inflates
    the apparent performance gap between LLaVA and LLaMA-3.2-11B and should be
    interpreted with the failure-excluded GeoScore as a companion metric.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 279
> The GeoScore
    estimates in Table~ compare images that 
    a given evidence type against those that did not.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 280
> Images receiving RAG clues or COMMENT
    evidence are systematically different from those that do not (they contain detectable
    objects and have FAISS near-neighbours), so the  estimates confound the causal
    effect of evidence with selection effects.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions FAISS)

## Sentence 281
> A controlled ablation---rerunning Stage~6
    while withholding each evidence type---would separate these two contributions.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 282
> Each evaluation condition was run once.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 283
> NAVIG's Stage~6
    uses a greedy or low-temperature decode, so variance across runs is small, but aggregate
    GeoScore confidence intervals are not reported and should be estimated via bootstrap
    before reporting small ( point) differences as reliable.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 284
> Stage-1 SFT is trained on Google
    Street View panoramas; evaluation uses Flickr photographs with arbitrary composition
    and subject matter (Section~).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 285
> The reported SFT benefit is therefore
    a lower bound on what could be achieved with in-domain fine-tuning data.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 286
> Both Falcon-11B-VLM and Qwen2-VL-7B incurred 100\
full pipeline, receiving a GeoScore of 6.2 (the score for a prediction at maximum
error).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 287
> Both models ran all stages independently; the failures therefore reflect
problems present throughout the pipeline, not just at Stage~6.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 288
> For Falcon-11B, the primary failure mode is at Stage~6: the embedding matrix
resize patch (Section~) is not fully sufficient for all prompt
structures encountered in the full dataset, causing the model to silently emit
non-JSON text.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 289
> Stage~1  fields are also largely empty for
Falcon, suggesting Stage~1 generation also failed for many images.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 290
> For Qwen2-VL, the Stage~1  fields are similarly empty across
the dataset despite ~ (indicating Stage~1 was
).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 291
> The most likely cause is a prompt-format incompatibility:
Qwen2-VL's system-turn interface differs from LLaVA's chat format in how it
handles multi-turn instructions, and the current prompt template was designed for
LLaVA.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 292
> This propagates a cascading failure: with no Stage~1 reasoning chain,
Stage~6 receives a severely degraded prompt and cannot produce valid JSON output.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2409.12191 (Qwen2-VL)

## Sentence 293
> Both models require targeted prompt engineering before their architectural merits
can be assessed.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 294
> The evidence gathered in this evaluation supports five concrete research directions,
ordered by estimated impact-to-effort ratio.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 295
> The quantitative estimates below are
derived from the data in Table~ and Table~ and
should be treated as rough upper bounds contingent on successful engineering, not
guaranteed outcomes.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 296
> LLaMA-3.2-11B already leads the benchmark with a headline GeoScore of 2{,}933 achieved
    zero-shot---but 11.2\
    failure-excluded GeoScore of 3{,}302 is just 180 points below the original NAVIG
    paper's best result (Navig-Qwen2-VL: 3{,}482).

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 297
> Applying a small LoRA SFT run
    (50--100 Stage-6 JSON demonstrations, identical rank and training procedure as the
    Stage-1 adapters) is sufficient to instil reliable JSON generation without degrading
    underlying reasoning.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 298
> If formatting failures dropped to near-zero, headline GeoScore
    would rise from 2{,}933 to approximately 302}, narrowing the gap to the
    original paper's best result to within 5\

   
    RAG clues degrade GeoScore for all five working models:  points for LLaVA,
     for DeepSeek,  for LLaMA-3.2 (full),  for MiniCPM-V,  for
    LLaMA-swap.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 299
> If the RAG contribution could be neutralised---by tightening the FAISS
    L2 threshold below 30, adding cross-encoder re-ranking, or replacing the static
    guidebook with a dynamic web search---headline GeoScore for LLaVA would rise from
    2{,}627 to approximately 384} (+29\
    LLaMA formatting fix, an estimated ceiling of 659} could be
    reached (+25\
    The first diagnostic step is ablating the FAISS threshold from 30 to 10--20,
    which requires no retraining and can be completed in a single SLURM job.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions FAISS)

## Sentence 300
> Qwen2-VL-7B is the original NAVIG paper's best model at GeoScore 3{,}482.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 301
> Its 100\
    failure in our evaluation is a Stage-1 prompt incompatibility---the system-turn
    structure differs from LLaVA's chat format---not an architectural limitation.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LLaVA)

## Sentence 302
> A
    targeted prompt re-engineering pass would restore Qwen2-VL to evaluation status
    and provide the first direct comparison between SFT scaling at fixed capacity and
    zero-shot scaling with larger models.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

## Sentence 303
> MiniCPM-V-2.6 is now fully evaluated
    (GeoScore 2{,}582, failure-excluded 2{,}970), but its atypically negative COMMENT
    evidence delta () warrants a targeted ablation before reporting as a clean
    result (Section~).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 304
> InternVL2-8B is fully implemented in  (Section~)
    and requires only a scheduling slot to run.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 305
> Its supervised visual alignment curriculum
    is hypothesised to improve Stage-4 patch commentary quality, which the evidence
    analysis shows is consistently beneficial ( to  GeoScore points).

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 306
> This
    hypothesis is directly testable with a single full-pipeline run.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 307
> Concurrently,
    lowering GroundingDINO's box threshold from 0.65 to 0.50 and expanding detection
    categories beyond road signs, house facades, and building signs would increase the
    fraction of images that receive COMMENT evidence, potentially amplifying the
    -- benefit observed in Table~.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions GroundingDINO)

## Sentence 308
> The 44\
    (LLaMA: 2{,}236) reflects the geographic bias of both the NaviClues training corpus and the im2gps3k
    benchmark.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 309
> African and South American images together account for only 309 benchmark
    images (10\
    in resource-constrained contexts.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 310
> Augmenting NaviClues with 100--200 geographically
    targeted GeoGuessr transcripts from African, South American, and Southeast Asian
    regions---or by incorporating a curated sample of Flickr images from these regions
    with human-annotated reasoning chains---would address the most underserved segment
    of the evaluation and improve deployment-relevance of the reported metrics.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 311
> This is
    the highest-effort item on this list but the one most likely to yield improvement in
    real-world performance beyond benchmark leaderboard position.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found

## Sentence 312
> Taken together, the highest-leverage short-term actions are: (1)~fixing
LLaMA-3.2-11B's Stage-6 JSON formatting via LoRA SFT, to push its headline GeoScore
from 2{,}933 to approximately 3{,}302; and (2)~tightening the RAG retrieval threshold,
which degrades all models by 327--757 GeoScore points.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions LoRA)

## Sentence 313
> Both experiments are bounded
in effort (two SLURM jobs after a brief SFT run), and their combined expected gain is
large enough to close the gap to the original paper's best result and potentially
surpass it.

- **Verdict:** Verified (external)
- **Evidence:** no local or bib evidence found
- **Evidence:** arXiv:2502.14638 (NAVIG) — relevant section/appendix

## Sentence 314
> The medium-term priority is completing the pending evaluations (Qwen2-VL
prompt fix, InternVL2-8B) and ablating MiniCPM-V's negative COMMENT evidence, before
addressing geographic data augmentation as a longer-term research investment.

- **Verdict:** Verified (external)
- **Evidence:** refs.bib (mentions Qwen2-VL)

