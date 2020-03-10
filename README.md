# depth_estimation

COOL STUFF HERE PLS!!

# Using Docker

Remove all dangling images
    docker rmi -f $(docker images -q --filter "dangling=true")

Build dockerfile -
    docker build --rm -t <myImageName> .

Start container with a volume
    docker run --gpus all -ti --name depth_estimation -p 8888:8888 --rm -v /home/$USER:/home/ <myImage>

Enter existing container
    docker stop <myContainerName>
    docker start <myContainerName>
    docker exec -it  myfirst bash

Open terminal in VSCode container this <this may be different on you pc>
    ctrl + shift + ' or or ~ or @ or #