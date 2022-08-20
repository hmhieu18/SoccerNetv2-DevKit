from SoccerNet.Evaluation.ActionSpotting import evaluate
results = evaluate(SoccerNet_path="/content/TemporallyAwarePooling_Data/content/SoccerNet/content/TemporallyAwarePooling/SoccerNet_TemporallyAwarePooling",
                   Predictions_path="/content/SoccerNetv2-DevKit/Task1-ActionSpotting/TemporallyAwarePooling_audio_vgg_new_nms/models/NetVLAD-vgg-4/outputs_test",
                   split="test", version=2, prediction_file="results_spotting.json", metric="tight")

print("tight Average mAP: ", results["a_mAP"])
print("tight Average mAP per class: ", results["a_mAP_per_class"])
print("tight Average mAP visible: ", results["a_mAP_visible"])
print("tight Average mAP visible per class: ", results["a_mAP_per_class_visible"])
print("tight Average mAP unshown: ", results["a_mAP_unshown"])
print("tight Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])