# unset QEFF_TEST_PROFILE
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/embedding_models/test_models.py &> test-emb-2-single.txt
# export QEFF_TEST_PROFILE=tiny_model
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/embedding_models/test_models.py &> test-emb-2-tiny-single.txt
# # pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_embedding_models/test_models.py &> audio-emb.txt
# unset QEFF_TEST_PROFILE
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_models/test_models.py &> audio-up.txt
# export QEFF_TEST_PROFILE=tiny_model
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_models/test_models.py &> audio-tiny.txt
# unset QEFF_TEST_PROFILE
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_embedding_models/test_models.py &> audio-emb.txt
# export QEFF_TEST_PROFILE=tiny_model
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_models/test_models.py &> audio-tiny.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_embedding_models/test_models.py &> audio-emb-tiny.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/disaggregated/test_disagg_mode.py &> model_test.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/disaggregated/test_disagg_mode.py &> model-config_test.txt
# export QEFF_TEST_PROFILE=tiny_model
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/disaggregated/test_disagg_mode.py &> test_disagg_mode-tiny.txt
unset QEFF_TEST_PROFILE
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/disaggregated/test_disagg_mode.py &> test_disagg_mode-qwen.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/disaggregated/test_disagg_mode.py &> test_disagg_mode-chunked-prefill.txt
# export QEFF_TEST_PROFILE=tiny_model
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_models/test_models.py &> test-audio-tiny.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_embedding_models/test_models.py &> test-audio-emb-tiny.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/embedding_models/test_models.py &> test-text-emb-tiny.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/embedding_models/test_qwen3vl_embedding.py &> test-qwen3vl-emb-tiny.txt

unset QEFF_TEST_PROFILE
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_models/test_models.py &> test-audio.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_embedding_models/test_models.py &> test-audio-emb.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/embedding_models/test_models.py &> test-text-emb.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/embedding_models/test_qwen3vl_embedding.py &> test-qwen3vl-emb.txt  

# pytest -s -v /home/vishsent/efficient-transformers/tests/transformers/models/audio_embedding_models/test_models.py &> audio-debug.txt
# pytest -s -v /home/vishsent/efficient-transformers/tests/diffusers/test_flux.py &> flux-full-model.txt
# python /home/vishsent/efficient-transformers/wav2vec2-debug-2.py &> layerwise-comp.txt
pytest -s -v /home/vishsent/efficient-transformers/tests/diffusers/test_flux.py &> flux-full-model-config-2.txt
pytest -s -v /home/vishsent/efficient-transformers/tests/diffusers/test_wan.py &> wan-full-model-config.txt
pytest -s -v /home/vishsent/efficient-transformers/tests/diffusers/test_wan_i2v.py &> wan_i2v-full-model-config.txt
