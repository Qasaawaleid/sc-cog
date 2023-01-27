#!/bin/bash

# Enable to trace the script
# set -x

APISERVER=k8s.ord1.coreweave.com

if [ -z ${NAMESPACE} ] || [ -z ${TOKEN} ]; then
  echo "NAMESPACE and TOKEN variable must be set: TOKEN=<token> NAMESPACE=<namespace> ./$(basename $BASH_SOURCE)"
  exit 1
fi  

# -------------- HELPER FUNCTIONS-----------

# Function loops until expected condition is met. 
# The condition is checked by helper functions, passed as the second argument:
# - `expect_vs_stopped`, i.e. status=VirtualServerStopped and started==False
# - `expect_vs_running`, i.e. statrus=VirtualServerReady and started==True
function wait_until_vs_status() {
  local NAME=$1
  local STATUS_FUNC=$2

  while true;do
    local VS_STATUS=($(get_vs_status ${NAME}))
    echo "Status:${VS_STATUS[0]}, Started:${VS_STATUS[1]}"
    ${STATUS_FUNC} ${VS_STATUS[0]} ${VS_STATUS[1]}
    if [ $? == 0 ];then
      return
    fi
    sleep 2
  done
}

# Helper function that returns STATUS and STARTED values for the Virtual Server
function get_vs_status() {
  local NAME=$1
  local CURRENT_STATUS_ARRAY=($(curl "https://${APISERVER}/apis/virtualservers.coreweave.com/v1alpha1/namespaces/${NAMESPACE}/virtualservers/${NAME}" \
    --header "Authorization: Bearer $TOKEN" \
    --header "Accept: application/json;as=Table;g=meta.k8s.io;v=v1" \
    --insecure -s | jq '.rows[] | .cells[1], .cells[3]' | tr -d '"'))
  echo "${CURRENT_STATUS_ARRAY[0]} ${CURRENT_STATUS_ARRAY[1]}"
}


# Start VM
function start_vm() {
  local NAME=$1
  curl -X PUT "https://${APISERVER}/apis/subresources.kubevirt.io/v1/namespaces/${NAMESPACE}/virtualmachines/${NAME}/start" \
    --header "Authorization: Bearer $TOKEN" \
    --insecure -s
    return $?
}

# Helper function that returns 0 when  statrus=VirtualServerReady and started==True
function expect_vs_running() {
  local STATUS=$1
  local STARTED=$2
  if [ "${STATUS}" == "VirtualServerReady" ] && [ "${STARTED}" == "True" ];then
    return 0
  fi
  return 1
}

# Virtual Server name
VS_NAME=github-runner

echo -e "\n------------- CREATE VS -----------------------------------"
start_vm ${VS_NAME}

echo -e "\n------------- WAIT FOR VMI UNTIL RUNNING ------------------"
wait_until_vs_status ${VS_NAME} expect_vs_running

exit 0