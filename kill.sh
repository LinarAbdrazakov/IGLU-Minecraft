  
#!/bin/sh

# a useful script for cleaning up a killed rllib job

set +e 
echo "killing xvfb"
ps aux | grep xvfb  | grep -v grep | awk '{print $2}' | xargs kill -9
echo "killing java"
ps aux | grep java  | grep -v grep | awk '{print $2}' | xargs kill -9
echo "killing minerl"
ps aux | grep minerl_patched  | grep -v grep | awk '{print $2}' | xargs kill -9
echo "killing ray"
ps aux | grep ray | grep -v grep | awk '{print $2}' | xargs kill -9