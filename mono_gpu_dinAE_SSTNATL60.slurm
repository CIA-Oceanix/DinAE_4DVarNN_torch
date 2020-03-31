#!/bin/bash
#SBATCH --job-name=gpu_dinAE_SSTNATL60         # nom du job
#SBATCH --partition=gpu_p1          # partition GPU choisie
#SBATCH --ntasks=1                  # nombre de taches a reserver (=nombre de GPU ici)
#SBATCH -A yrf@gpu                  # ccount identifier
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de coeurs CPU par tache (un quart du noeud ici)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread        # hyperthreading desactive
#SBATCH --time=06:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=gpu_mono%j.out     # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.out      # nom du fichier d'erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load pytorch-gpu/py3/1.4.0

# echo des commandes lancees
set -x

# execution du code
cd ~/DinAE
conda activate oceanix
python main_ApplyDinAETorch_SSTNATL60.py