# a file for errors from the job
#SBATCH --error=water_example.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=00:30:00
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com

module load singularity
singularity shsingularity shell --nv --bind /scratch/jessegv/data:/data,/home/jessegv/water:/home /scratch/jessegv/images/water.img 

