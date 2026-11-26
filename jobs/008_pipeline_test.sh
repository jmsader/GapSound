# timeout: 4000
rm -rf /mnt/gapsound/results/v2-test
RUN=/mnt/gapsound/results/v2-test N_SESS=24 LOADS=0.5 SETTINGS="shared salted" DURATION=120 MIN_SESS=12 N_PROBE=4 \
  PRESSURE=0 bash /mnt/gapsound/v2/run_all.sh
