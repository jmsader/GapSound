# timeout: 900
cd /mnt/gapsound/v2
rm -rf /tmp/sim && python sim_test.py /tmp/sim && time python analyze2.py /tmp/sim --quick
