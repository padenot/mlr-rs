all:
	cargo build --target=armv7-unknown-linux-gnueabihf --release
	arm-bela-linux-gnueabihf-strip target/armv7-unknown-linux-gnueabihf/release/mlr-rs-bela
	rsync --progress -r ../mlr-samples/ root@bela.local:~
	scp target/armv7-unknown-linux-gnueabihf/release/mlr-rs-bela root@bela.local:~
