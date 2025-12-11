# cosmic_hdct

1. load java module in valar
    * `module load java/16.0.2`
    * You can verify Java is now visible:
        `which java`
        `readlink -f "$(which java)"`
    * You should see somethig like:  `/opt/.../java`.
    * If the output path is for example: `/opt/jdk-16.0.2/bin/java` then your JDK root is `/opt/jdk-16.0.2`

2. Set JCC_JDK to the JDK root
    * Using the example above:
        `export JCC_JDK=/opt/jdk-16.0.2`
    * General rule: JCC_JDK should be the directory that contains bin/java, include, lib, etc.
    * You can also add that export line to your ~/.bashrc or to your job script so it is always set:
        `echo 'export JCC_JDK=/opt/jdk-16.0.2' >> ~/.bashrc`


* Create and environment with the packages in the requirements.txt
* 