# semisupervised_timeseries_infogan

A tensorflow implementation of informative generative adversarial network (InfoGAN ) to one dimensional ( 1D ) time series data with a supervised loss function.
So it's called semisupervised Info GAN.

See InfoGAN model ([https://arxiv.org/abs/1606.03657](https://arxiv.org/abs/1606.03657) ) 

## Dependencies

1. tensorflow = 1.0.1 
1. [sugartensor](https://github.com/buriburisuri/sugartensor) = 1.0.0.2
1. matplotlib = 2.2.3

## Sample Data

You can use any csv formatted time series data as following.

<pre><code>
time,serie,target
1,11.1,0
2,12.2,2
3,13.0,1
4,14,0,2
     .
     .
     .
</code></pre>

This file should be saved at 'asset/data/serie_target.csv' before you train the network. 

## Training the network

Execute
<pre><code>
python train.py
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/run-MMDD-HHmm to monitor training process.

## Generating sample time series data
 
Execute
<pre><code>
python generate.py
</code></pre>
to generate sample time series data.  The graph image of generated time series data will be saved in the 'asset/train' directory.

## Generated time series data sample

Work in progress...
  

## Related resources

1. [timeseries_gan](https://github.com/buriburisuri/timeseries_gan)
1. [supervised_infogan](https://github.com/buriburisuri/supervised_infogan)


# Authors
Jean-Jerome Levy based on [buriburisuri](https://github.com/buriburisuri) work.