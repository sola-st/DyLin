for ((i=2; i<=30; i++)); do
    timeout 2h bash run_single_project.sh $i
    if [ $? -eq 124 ]
    then
	    echo "Timed out $1"
	    docker stop $(docker ps -q)
    fi
done
