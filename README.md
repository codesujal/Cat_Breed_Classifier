Cat Breeds (12 classes)
Abyssinian
Bengal
Birman
Bombay
British Shorthair
Egyptian Mau
Maine Coon
Persian
Ragdoll
Russian Blue
Siamese
Sphynx

Dog Breeds (25 classes)
American Bulldog
American Pit Bull Terrier
Basset Hound
Beagle
Boxer
Chihuahua
English Cocker Spaniel
English Setter
German Shorthaired
Great Pyrenees
Havanese
Japanese Chin
Keeshond
Leonberger
Miniature Pinscher
Newfoundland
Pomeranian
Pug
Saint Bernard
Samoyed
Scottish Terrier
Shiba Inu
Staffordshire Bull Terrier
Wheaten Terrier
Yorkshire Terrier


(catbc) red_crown@burno-lap:~/Documents/Project/Cat_Breed_Classifier$ python training_model.py 
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778068655.775684   14701 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1778068655.775854   14701 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
I0000 00:00:1778068655.810115   14701 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778068656.617731   14701 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1778068656.617967   14701 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
Total cat images: 2371
Cat breeds: 12
E0000 00:00:1778068660.513971   14701 cuda_executor.cc:1737] INTERNAL: CUDA Runtime error: Failed call to cudaGetRuntimeVersion: Error loading CUDA libraries. GPU will not be used.: Error loading CUDA libraries. GPU will not be used.
W0000 00:00:1778068660.514286   14736 cuda_executor.cc:1755] Failed to determine cuDNN version (Note that this is expected if the application doesn't link the cuDNN plugin): INTERNAL: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
W0000 00:00:1778068660.529293   14701 gpu_device.cc:2365] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
Epoch 1/5
60/60 ━━━━━━━━━━━━━━━━━━━━ 33s 522ms/step - accuracy: 0.7252 - loss: 0.8860 - val_accuracy: 0.6716 - val_loss: 0.8705
Epoch 2/5
60/60 ━━━━━━━━━━━━━━━━━━━━ 32s 528ms/step - accuracy: 0.9172 - loss: 0.2753 - val_accuracy: 0.8505 - val_loss: 0.4489
Epoch 3/5
60/60 ━━━━━━━━━━━━━━━━━━━━ 31s 517ms/step - accuracy: 0.9488 - loss: 0.1657 - val_accuracy: 0.7916 - val_loss: 0.5877
Epoch 4/5
60/60 ━━━━━━━━━━━━━━━━━━━━ 33s 548ms/step - accuracy: 0.9726 - loss: 0.1110 - val_accuracy: 0.8316 - val_loss: 0.5427
Epoch 5/5
60/60 ━━━━━━━━━━━━━━━━━━━━ 32s 533ms/step - accuracy: 0.9900 - loss: 0.0704 - val_accuracy: 0.8042 - val_loss: 0.5689
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Training complete ✔ Model saved!




(catbc) red_crown@burno-lap:~/Documents/Project/Cat_Breed_Classifier$ python training_model.py 
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778069187.438188   16280 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1778069187.438356   16280 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
I0000 00:00:1778069187.471676   16280 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778069188.259094   16280 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1778069188.259328   16280 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
Total cat images: 2371
Cat breeds: 12
E0000 00:00:1778069192.644221   16280 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
I0000 00:00:1778069192.644250   16280 cuda_diagnostics.cc:160] env: CUDA_VISIBLE_DEVICES="-1"
I0000 00:00:1778069192.644257   16280 cuda_diagnostics.cc:163] CUDA_VISIBLE_DEVICES is set to -1 - this hides all GPUs from CUDA
I0000 00:00:1778069192.644265   16280 cuda_diagnostics.cc:171] verbose logging is disabled. Rerun with verbose logging (usually --v=1 or --vmodule=cuda_diagnostics=1) to get more diagnostic output from this module
I0000 00:00:1778069192.644266   16280 cuda_diagnostics.cc:176] retrieving CUDA diagnostic information for host: burno-lap
I0000 00:00:1778069192.644269   16280 cuda_diagnostics.cc:183] hostname: burno-lap
I0000 00:00:1778069192.644318   16280 cuda_diagnostics.cc:190] libcuda reported version is: 580.142.0
I0000 00:00:1778069192.644332   16280 cuda_diagnostics.cc:194] kernel reported version is: 580.142.0
I0000 00:00:1778069192.644333   16280 cuda_diagnostics.cc:284] kernel version seems to match DSO: 580.142.0
Epoch 1/5
119/119 ━━━━━━━━━━━━━━━━━━━━ 29s 227ms/step - accuracy: 0.7400 - loss: 0.7746 - val_accuracy: 0.6611 - val_loss: 1.0750
Epoch 2/5
119/119 ━━━━━━━━━━━━━━━━━━━━ 27s 225ms/step - accuracy: 0.9167 - loss: 0.2496 - val_accuracy: 0.8484 - val_loss: 0.4526
Epoch 3/5
119/119 ━━━━━━━━━━━━━━━━━━━━ 27s 227ms/step - accuracy: 0.9457 - loss: 0.1617 - val_accuracy: 0.7811 - val_loss: 0.6436
Epoch 4/5
119/119 ━━━━━━━━━━━━━━━━━━━━ 28s 234ms/step - accuracy: 0.9752 - loss: 0.0948 - val_accuracy: 0.6968 - val_loss: 1.0073
Epoch 5/5
119/119 ━━━━━━━━━━━━━━━━━━━━ 28s 237ms/step - accuracy: 0.9852 - loss: 0.0653 - val_accuracy: 0.8000 - val_loss: 0.6539
Model saved ✔
Training complete ✔