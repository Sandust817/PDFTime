export CUDA_VISIBLE_DEVICES=0

model_name="PDFTime"

datasets=(
    "ChlorineConcentration" "EthanolLevel" "MiddlePhalanxOutlineCorrect"
)

for dataset in "${datasets[@]}"; do
    python -u run.py \
      --task_name classification \
      --is_training 1 \
      --root_path "/root/data/UCR/${dataset}/" \
      --model_id "${dataset}" \
      --model "${model_name}" \
      --data UCR \
      --e_layers 2 \
      --batch_size 32 \
      --dropout 0.2 \
      --d_model 128 \
      --des "Exp" \
      --itr 1 \
      --learning_rate 0.0001 \
      --train_epochs 150 \
      --patience 20
done

datasets=(
    "ACSF1" "Adiac" "AllGestureWiimoteY"
    "ArrowHead" "BME" "Beef" "BeetleFly" "BirdChicken"
    "CBF" "Car" "Chinatown" "CinCECGTorso" 
    "Coffee" "Computers" "CricketX" "CricketY" "CricketZ"
    "DiatomSizeReduction" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxOutlineCorrect" "DistalPhalanxTW"
    "DodgerLoopDay" "DodgerLoopGame" "DodgerLoopWeekend" "ECG200" "ECG5000"
    "ECGFiveDays" "EOGHorizontalSignal" "Earthquakes" "ElectricDevices" 
    "FaceAll" "FaceFour" "FacesUCR" "FiftyWords"
    "Fish" "FordA" "FordB" "FreezerRegularTrain" "FreezerSmallTrain"
    "Fungi" "GestureMidAirD1" "GestureMidAirD2" "GestureMidAirD3" "GesturePebbleZ1"
    "GesturePebbleZ2" "GunPoint" "GunPointAgeSpan" "GunPointMaleVersusFemale" "GunPointOldVersusYoung"
    "Ham" "HandOutlines" "Herring" "HouseTwenty"
    "InsectEPGRegularTrain" "InsectEPGSmallTrain" "InsectWingbeatSound" "ItalyPowerDemand"
    "LargeKitchenAppliances" "Lightning2" "Lightning7" "Mallat" "Meat"
    "MedicalImages" "MelbournePedestrian" "MiddlePhalanxOutlineAgeGroup"
    "MixedShapesRegularTrain" "MixedShapesSmallTrain" "MoteStrain" "NonInvasiveFetalECGThorax1" "NonInvasiveFetalECGThorax2"
    "OSULeaf" "PickupGestureWiimoteZ" "PigCVP" "Plane"
    "PowerCons" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxTW" "RefrigerationDevices"
    "Rock" "ScreenType" "SemgHandGenderCh2" "SemgHandMovementCh2" "SemgHandSubjectCh2"
    "ShakeGestureWiimoteZ" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances" "SmoothSubspace"
    "SonyAIBORobotSurface1"
    "StarLightCurves" "Strawberry" "SwedishLeaf"
    "Symbols" "SyntheticControl" "ToeSegmentation1" "ToeSegmentation2" "Trace"
    "TwoLeadECG" "TwoPatterns" "UMD" "UWaveGestureLibraryAll" "UWaveGestureLibraryX"
    "UWaveGestureLibraryY" "UWaveGestureLibraryZ" "Wafer" "Wine" "WordSynonyms"
    "Worms" "WormsTwoClass"
)

for dataset in "${datasets[@]}"; do
    python -u run.py \
      --task_name classification \
      --is_training 1 \
      --root_path "/root/data/UCR/${dataset}/" \
      --model_id "${dataset}" \
      --model "${model_name}" \
      --data UCR \
      --e_layers 2 \
      --batch_size 32 \
      --dropout 0.2 \
      --d_model 128 \
      --des "Exp" \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 150 \
      --patience 20
done
datasets=(
    "AllGestureWiimoteX" "AllGestureWiimoteZ"
     "Crop" "EOGVerticalSignal" "Haptics" "InlineSkate" "MiddlePhalanxTW"
    "OliveOil" "PLAID" "PhalangesOutlinesCorrect" "Phoneme"
    "PigAirwayPressure" "PigArtPressure" "SonyAIBORobotSurface2" "Yoga"
)

for dataset in "${datasets[@]}"; do
    python -u run.py \
      --task_name classification \
      --is_training 1 \
      --root_path "/root/data/UCR/${dataset}/" \
      --model_id "${dataset}" \
      --model "${model_name}" \
      --data UCR \
      --e_layers 2 \
      --batch_size 64 \
      --dropout 0.2 \
      --d_model 128 \
      --des "Exp" \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 150 \
      --patience 20
done
