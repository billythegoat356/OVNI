FFMPEG Libraries present in this package were built using the following configure options

x64:   ./configure --enable-shared --disable-encoders --disable-decoders --enable-decoder=vp9 --toolchain=msvc
win32: ./configure --enable-shared --disable-encoders --disable-decoders --enable-decoder=vp9 --toolchain=msvc --arch=x86
x86_64: LDSOFLAGS=-Wl,-rpath,\''$$$$ORIGIN'\'  ./configure --enable-shared --disable-encoders --disable-decoders --enable-decoder=vp9  --arch=x86_64

Only header files and libraries that are required for compiling the sample applications inside the package have been included.



