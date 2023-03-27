By Simon Plovyt @splovyt
Published May 22, 2019

------------------------------------------------------------------------

## Learning objective

This tutorial explains how to use a Kubernetes system to deploy a cloud
instance for your Model Asset Exchange (MAX) model. The Kubernetes
service lets you easily scale the containerized model from supporting a
low rate of queries to production-level queries. The free&nbsp;[IBM
Cloud Kubernetes
Service](https://web.archive.org/web/20210419215534/https://www.ibm.com/cloud/container-service)&nbsp;is
limited to 1 worker, but this works fine for this application.
Let&rsquo;s get started!

## Prerequisites

If you are not familiar with the Model Asset Exchange,
this&nbsp;[introductory
article](https://web.archive.org/web/20210419215534/https://developer.ibm.com/articles/introduction-to-the-model-asset-exchange-on-ibm-developer/)&nbsp;provides
a good overview.

Additionally, you need the following:

- An&nbsp;[IBM Cloud
  account](https://web.archive.org/web/20210419215534/http://ibm.biz/max-contents).
  You can sign up for a free account.
- An available Kubernetes cluster on your IBM Cloud account. The free
  cluster type will suffice. Create a Kubernetes cluster from
  the&nbsp;[IBM Cloud
  catalog](https://web.archive.org/web/20210419215534/https://cloud.ibm.com/kubernetes/catalog/cluster?cm_sp=ibmdev-_-developer-tutorials-_-cloudreg).
- The IBM Cloud command-line interface (CLI) installed. After creating a
  Cloud account and a Kubernetes cluster, you must communicate with the
  cluster using your local machine.&nbsp;[Installation
  instructions](https://web.archive.org/web/20210419215534/https://cloud.ibm.com/docs/cli/reference/ibmcloud?topic=cloud-cli-ibmcloud-cli#ibmcloud-cli)&nbsp;are
  available, or you can perform a web search for &lsquo;IBM Cloud
  CLI.&rsquo;

## Estimated time

With the prerequisites available, it should take you approximately 20
minutes to complete this tutorial.

## Steps

This tutorial consists of the following steps:

1.  Accessing the Kubernetes cluster
2.  Deploying the MAX model

A troubleshooting section can be found at the end of this tutorial.

### Step 1. Accessing the Kubernetes cluster

With an IBM Cloud account available, a Kubernetes cluster created, and
the Cloud CLI installed
(see&nbsp;[Prerequisites](https://web.archive.org/web/20210419215534/https://developer.ibm.com/tutorials/deploy-max-models-to-ibm-cloud-with-kubernetes/#prerequisites)),
the actual deployment step itself is the easiest part. Begin by
establishing the connection between your local machine and the
Kubernetes cluster.

Use the CLI to log in to IBM Cloud on your local machine and specify
what cluster you are targeting in the cloud. All of these instructions
are also available under the&nbsp;**Access tab**&nbsp;on the online page
of your Kubernetes cluster. Go to this&nbsp;**Access tab**&nbsp;to
ensure that you are working with the most up-to-date instructions.

![IBM cloud
access](https://web.archive.org/web/20210419215534im_/https://developer.ibm.com/developer/default/tutorials/deploy-max-models-to-ibm-cloud-with-kubernetes/images/ibm-cloud-access.png)

Now, execute the instructions.

1.  Log in to the cluster from your local machine.

    ```
     ibmcloud login -a https://cloud.ibm.com
    ```

  

2.  Set the IBM Cloud region to the Kubernetes Service region that you
    selected when creating your cluster. The example cluster is in US
    South, but you can find information on
    other&nbsp;[regions](https://web.archive.org/web/20210419215534/https://cloud.ibm.com/docs/containers?topic=containers-regions-and-zones).

    ```
     ibmcloud ks region-set us-south
    ```

  

3.  Get the command to set the environment variable and download the
    Kubernetes configuration files to the cluster.

    > Note: Replace the last argument (`max-deployment-cluster`) with
    > the name you gave your Kubernetes cluster.

    ```
     ibmcloud ks cluster-config max-deployment-cluster
    ```

  

4.  Copy the output from the previous command and paste it in your
    terminal. The command starts with&nbsp;`export KUBECONFIG=...`. This
    command must be run because it sets the
    required&nbsp;`KUBECONFIG`&nbsp;environment variable.

5.  Verify that you can connect to your cluster by listing your worker
    nodes. The&nbsp;`kubectl`&nbsp;commands are part of the Kubernetes
    CLI, which was installed in the same process as the IBM Cloud CLI.
    If the&nbsp;`kubectl`&nbsp;command returns&nbsp;`command not found`,
    you must install the Kubernetes CLI plug-in separately. More
    information on installing the plug-in separately can be found in the
    Troubleshooting section at the end of this tutorial.

    ```
    kubectl get nodes
    ```

  

You are now ready for the final step of the deployment.

### Step 2. Deploying the MAX model

The actual deployment process is short. Now that the connection with the
remote Kubernetes cluster has been set up, you only have to apply the
model&rsquo;s configuration file to the Kubernetes cluster to get the
API up and running.

Let&rsquo;s download the model&rsquo;s repository, which contains the
YAML configuration file.

1.  Clone the MAX-Model repository from GitHub. The following code shows
    how to do this for the MAX-Object-Detector, but you can
    replace&nbsp;`MAX-Object-Detector`&nbsp;with whichever MAX Model you
    want to use.

    ```
     git clone https://github.com/IBM/MAX-Object-Detector.git
    ```

  

2.  Find the .yaml file in the downloaded directory.

    ```
     cd MAX-Object-Detector

     ls *.yaml
    ```

  

3.  Apply the configuration file to your Kubernetes cluster. Again,
    replace the&nbsp;`max-object-detector.yaml`&nbsp;with the .yaml file
    corresponding to your model. The following command deploys the
    model.

    ```
     kubectl apply -f ./max-object-detector.yaml
    ```

  

4.  After the model is deployed (give it some time), find the public IP
    address and port that the API is served on.

    - The public IP

      ```
        ibmcloud cs workers mycluster
      ```

    

    - The port

      > Note: Replace&nbsp;`max-object-detector`&nbsp;with the service
      > name corresponding to your model.

      ```
        kubectl describe service max-object-detector | grep NodePort
      ```

    

5.  Navigate to the&nbsp;`http://<PUBLIC_IP>:<PORT>`&nbsp;address in
    your browser. You should find the API front end as shown in the
    following image.

    ![MAX object detector
    api](https://web.archive.org/web/20210419215534im_/https://developer.ibm.com/developer/default/tutorials/deploy-max-models-to-ibm-cloud-with-kubernetes/images/object-detector-api.png)

This means that the MAX model has been successfully deployed.
Congratulations!

##### Some useful commands

Kubernetes CLI:

```
# Show the log of all events
kubectl get events

# Show all pods
kubectl get pods

# Show all services
kubectl get services

# Show all deployments
kubectl get deployments

# Show the details of a service
kubectl describe service SERVICE_NAME

# Show the details of a node
kubectl describe node NODE_NAME

# Delete a service
kubectl delete services SERVICE_NAME

# Delete a deployment
kubectl delete deployment DEPLOYMENT_NAME

# Debugging a pod
kubectl logs POD_NAME
```

IBM Cloud CLI:

```
# Show all clusters
ibmcloud cs clusters

# Show the workers of a specific cluster
ibmcloud cs workers CLUSTER_NAME
```


##### Further scaling

Scaling a Kubernetes-powered application beyond the 2vCPU and 4 GB RAM
specifications of the free cluster is as easy as clicking a button.
Adding worker nodes (computation units) or specifying advanced cluster
parameters (for example, more CPU or RAM) can be done easily from the
Cloud Dashboard. However, you must&nbsp;[upgrade to a standard
cluster](https://web.archive.org/web/20210419215534/https://cloud.ibm.com/docs/containers?topic=containers-cs_ov#cluster_types)&nbsp;to
access these.

Additionally, with more worker nodes, a single node&rsquo;s public IP
address does not suffice. Therefore, services
like&nbsp;[Ingress](https://web.archive.org/web/20210419215534/https://kubernetes.io/docs/concepts/services-networking/ingress/)&nbsp;and&nbsp;[Loadbalancer](https://web.archive.org/web/20210419215534/https://kubernetes.io/docs/concepts/services-networking/ingress/)&nbsp;must
be added as well. Exposing a public IP address on the deployment level
instead of on the node level with the Loadbalancer can be set up in a
few easy steps as demonstrated below.

1.  Show the existing deployments and note the deployment name.

    ```
     kubectl get deployments
    ```

  

2.  With the deployment name, you can expose this deployment to a public
    IP address and port by using the following command.
    Replace&nbsp;`<DEPLOYMENT_NAME>`&nbsp;with the name of the
    deployment, and replace&nbsp;`<SERVICE_NAME>`&nbsp;with a custom
    name for your microservice. The&nbsp;`target-port`&nbsp;flag points
    to the port of the running container, which is set to 5000 by
    default for MAX models. The port that you want to expose on the
    public IP address can be set with the&nbsp;`port`&nbsp;flag. Here,
    it&rsquo;s specified as&nbsp;`80`.

    ```
     kubectl expose deployment <DEPLOYMENT_NAME> --port=80 --target-port=5000 --name=<SERVICE_NAME> --type=LoadBalancer --load-balancer-ip=''
    ```

  

3.  The public IP and port are now being configured as part of the
    service. When done (it can be a couple of minutes), the external IP
    address shows up in the output of the following command.

    ```
     kubectl get service --watch
    ```

  

4.  You can verify that your service is up and running by typing the IP
    address and port in the browser like&nbsp;`http://<IP>:<PORT>`.

More information can be found in the&nbsp;[Deploy a MAX model-serving
microservice on IBM Cloud Kubernetes
Service](https://web.archive.org/web/20210419215534/https://github.com/CODAIT/MAX-cloud-deployment-cheatsheets/tree/master/ibm-cloud)&nbsp;GitHub
repository.

## Troubleshooting

An easy way to detect general issues with your Kubernetes cluster is to
visit the Kubernetes Dashboard on IBM Cloud. Any critical or pending
worker nodes are highlighted. In addition, you can run the following
commands in the terminal.

- Check Kubernetes events

  ```
    kubectl get events
  ```



- Get the logs for a pod

  ```
    kubectl logs POD_NAME
  ```



##### Memory issues and killed containers

When trying to deploy a model with above-average memory requirements
(for example, the MAX-Image-Resolution-Enhancer), you will run into
a&nbsp;`TypeError: Failed to fetch`&nbsp;error message. Although the
front end is still responsive, this error often means that the container
was killed and restarted due to exceeding the memory limit after posting
an input image. In this case, you must upgrade to a standard Kubernetes
cluster to choose the specifications of the worker nodes. Increasing the
resources of the node solves this issue.

##### Can&rsquo;t find the kubectl command

Most likely, this means that you are working with an older version of
the IBM Cloud CLI for which the Kubernetes CLI was not included in the
package. Verify that
the&nbsp;`container-service/kubernetes-service`&nbsp;plug-in is not
present by running the&nbsp;`ibmcloud plugin list`&nbsp;command. Run
the&nbsp;`ibmcloud plugin install container-service`&nbsp;command to
install the plug-in,
or&nbsp;[reinstall](https://web.archive.org/web/20210419215534/https://cloud.ibm.com/docs/cli?topic=cloud-cli-uninstall-ibmcloud-cli)&nbsp;the
IBM Cloud CLI. You can find more information on the&nbsp;[official
Kubernetes
page](https://web.archive.org/web/20210419215534/https://kubernetes.io/docs/tasks/tools/install-kubectl/)&nbsp;or
on the&nbsp;[IBM Cloud CLI
page](https://web.archive.org/web/20210419215534/https://cloud.ibm.com/docs/containers?topic=containers-cs_cli_install#cs_cli_install_steps).

## Summary

This tutorial illustrates how to deploy a model container from the Model
Asset Exchange to IBM Cloud using Kubernetes by following two easy
steps: accessing the Kubernetes cluster and deploying the MAX model.

Collections[\#Open Source Data & AI
Technologies](https://web.archive.org/web/20210419215534/https://developer.ibm.com/collections/codait/)
