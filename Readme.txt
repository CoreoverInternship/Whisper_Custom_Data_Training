This project prepairs a custom data set, and finetunes a whisper model on custom local data. We specifically used hindi call data to finetune the model to better understand and transcribe the hindi language when spoken on a call. The purpouse of this is to crate speech to text of the hindi language when on a call through zoom or teams.

To use this you need to unzip Custom_Training_Set.zip this should make a folder with many folders in it, each folder should have multiple audio clips and a text file with the  transcripts of each audio clip.
Run DataPrep.py with the correct directory of the custom trainig set and this should make training and testing csv files wich should include the directory of the audio clip, and the proper translation.
Now you  are ready to start training. ensure file paths are correct in customDataTraining.py and then run it, it should use the CSV files created to finetune The model on the custom data.

Please note these scripts where used on a multi-GPU device so this is ment to run with an accelerate config. to run type the following command into terminal "accelerate launch customDataTraining.py" and to configure accelerate config type "accelerate config"
