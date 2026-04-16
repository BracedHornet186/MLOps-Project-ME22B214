# %%
!pip config set global.disable-pip-version-check true

# %%
!cp /kaggle/input/glomap/glomap_build/glomap /usr/local/bin/
!chmod 755 /usr/local/bin/glomap

!cp -r /kaggle/input/glomap/glomap_build glomap_lib
!rm glomap_lib/librt.so.1  # conflict
!rm glomap_lib/libdl.so.2
!rm glomap_lib/libnvJitLink.so.12*

# %%
!cp -r /kaggle/input/ns64-imc2025lib/lib_py311_t4/custom_ops ./
!cp /kaggle/input/ns64-imc2025lib/lib_py311_t4/score_computation_cuda.cpython-311-x86_64-linux-gnu.so ./
!cp /kaggle/input/ns64-imc2025lib/lib_py311_t4/value_aggregation_cuda.cpython-311-x86_64-linux-gnu.so ./

# %%
!pip install --no-deps /kaggle/input/ns64-imc2025lib/ns64_imc2025lib-0.1.50-py3-none-any.whl

# %%
!pip install --no-deps /kaggle/input/ns64-imc2025lib/addict-2.4.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/asmk-0.1-cp311-cp311-linux_x86_64.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/attrs-25.3.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/cholespy-2.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/croco-0.1.2-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/curope-0.0.0-cp311-cp311-linux_x86_64.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/dad-0.2.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/dedode-0.0.1-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/dotmap-1.3.30-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/dsine-0.0.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/dust3r-0.1.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/e2cnn-0.2.3-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/einops-0.8.1-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/faiss_cpu-1.10.0-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/flask-3.1.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/glcontext-3.0.0-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/hloc-1.5-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/HTML4Vision-0.5.0-py2.py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/huggingface_hub-0.30.2-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/hydra_core-1.3.2-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/iglovikov_helper_functions-0.0.53-py2.py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/kornia_moons-0.2.9-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/lightglue-0.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/lightning-2.5.1-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/loguru-0.7.3-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/mast3r-0.1.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/mmengine-0.10.7-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/moderngl-5.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/moge-1.0.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/mono-0.0.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/mpsfm-0.1.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/plyfile-1.1-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/pyceres-2.4-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/pycolmap-3.11.1-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/superglue_pretrained_network-0.0.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/utils3d-0.0.2-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/vggt-0.0.1-py3-none-any.whl
!pip install --no-deps /kaggle/input/ns64-imc2025lib/yacs-0.1.8-py3-none-any.whl

# %%
!mkdir -p hf_grounding_dino_base
!mkdir -p hf_sam_vit_base
!mkdir -p hf_vggt_1b
!mkdir -p siglip2_so400m_patch14_384

!ln -s /kaggle/input/imc2023models/grounded_sam/models--IDEA-Research--grounding-dino-base/blobs/5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21 ./hf_grounding_dino_base/model.safetensors
!ln -s /kaggle/input/imc2023models/grounded_sam/models--IDEA-Research--grounding-dino-base/blobs/5a7f6206a1e488c54316e1f594311dd47a03a41b ./hf_grounding_dino_base/config.json
!ln -s /kaggle/input/imc2023models/grounded_sam/models--IDEA-Research--grounding-dino-base/blobs/5cb45d963917ed130ce46a93204b349cbec21131 ./hf_grounding_dino_base/preprocessor_config.json
!ln -s /kaggle/input/imc2023models/grounded_sam/models--IDEA-Research--grounding-dino-base/blobs/688882a79f44442ddc1f60d70334a7ff5df0fb47 ./hf_grounding_dino_base/tokenizer.json
!ln -s /kaggle/input/imc2023models/grounded_sam/models--IDEA-Research--grounding-dino-base/blobs/a8b3208c2884c4efb86e49300fdd3dc877220cdf ./hf_grounding_dino_base/special_tokens_map.json
!ln -s /kaggle/input/imc2023models/grounded_sam/models--IDEA-Research--grounding-dino-base/blobs/ed97a84add5f9b2091e756765ad3ba087a345e17 ./hf_grounding_dino_base/tokenizer_config.json
!ln -s /kaggle/input/imc2023models/grounded_sam/models--IDEA-Research--grounding-dino-base/blobs/fb140275c155a9c7c5a3b3e0e77a9e839594a938 ./hf_grounding_dino_base/vocab.txt

!ln -s /kaggle/input/imc2023models/grounded_sam/models--facebook--sam-vit-base/blobs/5880e4f874dbeb2a1921c9cf3bb44da529e46bf4 ./hf_sam_vit_base/config.json
!ln -s /kaggle/input/imc2023models/grounded_sam/models--facebook--sam-vit-base/blobs/732fbaf0c512b97d8d9161f51bc157bfb2873d12 ./hf_sam_vit_base/preprocessor_config.json
!ln -s /kaggle/input/imc2023models/grounded_sam/models--facebook--sam-vit-base/blobs/892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579 ./hf_sam_vit_base/model.safetensors

!ln -s /kaggle/input/imc2025models/models--facebook--VGGT-1B/blobs/303bf21400e2723e8ff9c0c7ceb6d86859b1ddeb ./hf_vggt_1b/config.json
!ln -s /kaggle/input/imc2025models/models--facebook--VGGT-1B/blobs/f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e ./hf_vggt_1b/model.safetensors

!ln -s /kaggle/input/imc2025models/models--google--siglip2-so400m-patch14-384/blobs/7c1e0ed1759922fc4eb362d3b405958e829f364d ./siglip2_so400m_patch14_384/config.json
!ln -s /kaggle/input/imc2025models/models--google--siglip2-so400m-patch14-384/blobs/e9e084ab5a0d74573432f1dcf11c1bdd8d9b3655 ./siglip2_so400m_patch14_384/preprocessor_config.json
!ln -s /kaggle/input/imc2025models/models--google--siglip2-so400m-patch14-384/blobs/9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2 ./siglip2_so400m_patch14_384/model.safetensors

# %%
import numpy as np
import yaml
import os
import sys
sys.path.insert(0, '/usr/local/lib/python3.11/dist-packages/ns64_imc2025lib')
os.environ["OPEN3D_DISABLE_WEB_VISUALIZER"] = "true"
os.environ['DEFAULT_DATASET_DIR'] = '/kaggle/input/image-matching-challenge-2025'
os.environ['DEFAULT_TMP_DIR'] = '/kaggle/tmp'
os.environ['DEFAULT_MODEL_LIST_PATH'] = '/kaggle/input/ns64-imc2025lib/models.yaml'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
#os.environ['DEFAULT_MODEL_LIST_PATH'] = '/kaggle/input/imc2023models/models.yaml'
#os.environ['SCENE_SPACE_DIR_PERSISTENT'] = 'yes'

# %%
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.use_deterministic_algorithms(True)

# %%
from ns64_imc2025lib.config import SubmissionConfig
from ns64_imc2025lib.kernel import run_and_save_submission
from ns64_imc2025lib.data import on_kaggle_kernel_rerun

# %%
conf = SubmissionConfig.load_config_from_pipeline_config_string("""
# conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-022-b.yaml
# ------------------------------------
# ------------------------------------
type: imc2025

imc2025_pipeline:
  point_tracking_matchers:
    - type: mast3r_hybrid
      impl_version: v2
      local_features:
        - type: lightglue_aliked
          lightglue_aliked:
            weight_path: ALIKED_LIGHTGLUE_N16
            max_num_keypoints: 4096
          resize:
            func: lightglue
            lg_resize: 1280
        - type: magicleap_superpoint
          magicleap_superpoint:
            weight_path: MAGICLEAP_SUPERPOINT
            nms_radius: 4
            keypoint_threshold: 0.0005
            max_keypoints: 4096
            remove_borders: 4
            fix_sampling: true
          resize:
            func: magicleap
            ml_resize: 1600
      mast3r_hybrid:
        size: 512
        dense_min_matches: 15
        dense_subsample: 8
        dense_pixel_tol: 5
        dense_match_threshold: 1.001
        dense_match_topk: null  # Use all
        sparse_min_matches: 15
        sparse_match_threshold: 1.001
        sparse_match_topk: null  
        model:
          use_amp: true
          weight_path: MAST3R

  shortlist_generator: 
    type: ensemble
    ensemble:
      all_pairs_fallback_threshold: 0
      shortlist_generators:
        - type: mast3r_retrieval_asmk
          mast3r_retrieval_asmk_fallback_threshold: 0
          mast3r_retrieval_asmk_remove_swapped_pairs: true
          mast3r_retrieval_asmk_make_pairs_fps_n: 10
          mast3r_retrieval_asmk_make_pairs_fps_k: 25
          mast3r_retrieval_asmk_make_pairs_fps_dist_threshold: null
          mast3r_retrieval_asmk:
            mast3r:
              weight_path: MAST3R
              retrieval_weight_path: MAST3R_RETRIEVAL
              retrieval_codebook_path: MAST3R_RETRIEVAL_CODEBOOK
        - type: global_desc
          global_desc_model: mast3r_retrieval_spoc
          global_desc_batch_size: 1
          global_desc_num_workers: 1
          global_desc_similar_distance_threshold: 9999
          global_desc_topk: 10
          global_desc_fallback_threshold: 0
          global_desc_remove_swapped_pairs: true
          global_desc_num_refills_when_no_matches: 10
          mast3r_retrieval_spoc:
            mast3r:
              weight_path: MAST3R
              retrieval_weight_path: MAST3R_RETRIEVAL
              retrieval_codebook_path: MAST3R_RETRIEVAL_CODEBOOK
            global_desc_type: retrieval_spoc
        - type: global_desc
          global_desc_model: dinov2
          global_desc_batch_size: 1
          global_desc_num_workers: 1
          global_desc_similar_distance_threshold: 9999
          global_desc_topk: 10
          global_desc_fallback_threshold: 0
          global_desc_remove_swapped_pairs: true
          global_desc_num_refills_when_no_matches: 10
          dinov2:
            pretrained_model: DINOV2_BASE
        - type: global_desc
          global_desc_model: isc
          global_desc_batch_size: 1
          global_desc_num_workers: 1
          global_desc_similar_distance_threshold: 9999
          global_desc_topk: 10
          global_desc_fallback_threshold: 0
          global_desc_remove_swapped_pairs: true
          global_desc_num_refills_when_no_matches: 10
          isc:
            weight_path: ISC
  
  reconstruction:
    fill_zero_Rt: false
    fill_nan_Rt: false
    fill_nearest_position: false
    mapper_min_model_size: 3
    mapper_max_num_models: 25
  
  clustering: null
""")
print(conf.model_dump_json(indent=4))

with open("config.yaml", "w") as fp:
    yaml.safe_dump(conf.model_dump(), fp)

# %%
!PYTHONPATH=/usr/local/lib/python3.11/dist-packages/ns64_imc2025lib:$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 LD_LIBRARY_PATH=./glomap_lib:$LD_LIBRARY_PATH \
 OPEN3D_DISABLE_WEB_VISUALIZER=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DEFAULT_DATASET_DIR=/kaggle/input/image-matching-challenge-2025 \
 CUBLAS_WORKSPACE_CONFIG=':4096:8' \
 DEFAULT_TMP_DIR=/kaggle/tmp DEFAULT_MODEL_LIST_PATH=/kaggle/input/ns64-imc2025lib/models.yaml \
 torchrun --nnodes 1 --nproc_per_node 2 --standalone -m ns64_imc2025lib.kernel -c config.yaml --env-name kernel --dist --kaggle-submit

# %%
RUN_SINGLE_PROCESS = False
if RUN_SINGLE_PROCESS:
    print("RUN_SINGLE_PROCESS=True")
    if on_kaggle_kernel_rerun():
        run_and_save_submission(conf)
    else:
        conf.target_data_type = 'submission-fast-commit'
        run_and_save_submission(conf)

# %%
!ls submission.csv

# %%



