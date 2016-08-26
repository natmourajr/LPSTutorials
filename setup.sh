# LPSTutorials Setup Script
#
# Author: natmourajr@gmail.com
#

# Env Variables
export MY_PATH=$PWD

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Ubuntu
    export WORKSPACE=$MY_PATH
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    export WORKSPACE=$MY_PATH
    # For matplotlib
	export LC_ALL=en_US.UTF-8
	export LANG=en_US.UTF-8
fi

export RESULTSPATH=$WORKSPACE/Results

# Folder Configuration
if [ -d "$RESULTSPATH" ]; then
    read -e -p "Folder $RESULTSPATH exist, Do you want to erase it? [Y,n] " yn_erase
	if [ "$yn_erase" = "Y" ]; then
        echo "creating RESULTSPATH struct"
        rm -rf $RESULTSPATH
        mkdir $RESULTSPATH
        cd $WORKSPACE/Packages
        for i in $(ls -d */); do 
        	mkdir $RESULTSPATH/${i%%/};
        	mkdir $RESULTSPATH/${i%%/}/output_files;
        	mkdir $RESULTSPATH/${i%%/}/picts; 
        done
    fi
    
else
    echo "RESULTSPATH: $RESULTSPATH doesnt exists"
    echo "creating RESULTSPATH struct"
    rm -rf $RESULTSPATH
    mkdir $RESULTSPATH
    cd $WORKSPACE/Packages
    for i in $(ls -d */); do 
    	mkdir $RESULTSPATH/${i%%/};
    	mkdir $RESULTSPATH/${i%%/}/output_files;
        mkdir $RESULTSPATH/${i%%/}/picts;
    done
fi

cd $MY_PATH