This project prepares a custom data set and finetunes a whisper model on custom local data. We specifically used Hindi call data to finetune the model to better understand and transcribe the Hindi language when spoken on a call. The purpose of this is to convert speech-to-text the Hindi language when on a call through Zoom or Teams.

To use this you need to unzip Custom_Training_Set.zip this should give you a folder with many folders in it. Each folder should have multiple audio clips and a text file with the transcript of each audio clip.

Run DataPrep.py with the correct directory of the custom training set and this should make training and testing CSV files which should include the directory of the audio clip and the corresponding text.

Now, you are ready to start training. Ensure the file paths are correct in customDataTraining.py and then run it, it should use the CSV files to finetune the model on the custom data.

Please note that these scripts were used on a multi-GPU device so this is meant to run with an accelerate config. To run it with an accelerate config, type the following command into the terminal "accelerate launch customDataTraining.py" and to configure the accelerate config type "accelerate config".
