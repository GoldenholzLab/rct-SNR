
str="test_folder_";
for ((i=1; i<10; i=i+1)); 
do
    folder_name="$str$i"
    if [ ! -d $folder_name ]; then
        mkdir $folder_name;
    else
        echo "$folder_name already exists"
    fi;
done;