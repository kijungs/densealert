rm DenseAlert-1.0.tar.gz
rm -rf DenseAlert-1.0
mkdir DenseAlert-1.0
cp -R ./{compile.sh,package.sh,src,Makefile,README.txt,*.jar,example_data.txt.gz,user_guide.pdf} ./DenseAlert-1.0
tar cvzf DenseAlert-1.0.tar.gz --exclude='._*' ./DenseAlert-1.0
rm -rf DenseAlert-1.0
echo done.
