docker stop emu1 emu2 emu-gw emu3 emu4 emu5 emu6 emu7 emu8 emu9 emu10 emu11 emu12 emu13 emu14 emu15

docker rm emu1 emu2 emu-gw emu3 emu4 emu5 emu6 emu7 emu8 emu9 emu10 emu11 emu12 emu13 emu14 emu15

docker compose up -d

for i in {1..15}
do
 docker exec -i emu$i /opt/android/platform-tools/adb shell settings put system pointer_location 0 --user 0
done
