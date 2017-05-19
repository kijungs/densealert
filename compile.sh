echo compiling java sources...
rm -rf class
mkdir class

javac -d class $(find ./src -name *.java)

echo make jar archive...
cd class
jar cf DenseAlert-1.0.jar ./
rm ../DenseAlert-1.0.jar
mv DenseAlert-1.0.jar ../
cd ..
rm -rf class

echo done.
