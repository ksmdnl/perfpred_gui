
<h1>Performance estimation inference</h1>
<p>This is an attempt to deploy the performance prediction on real life videos</p>
<h2>TODOs:</h2>
<p>
we can get the PSNR but what do we need to estimate the mIoU?

some ideas:
<ul>
    <li>
        get the regression coefficient and just calculate the estimated mIoU from that
    </li>
    <li>
        using bayes inference or kalman filter
    </li>
</ul>

bayes inference or kalman filter:
<ul>
    <li>
        transform the PSNR and the mIoU from the val data into a probability density. we need:
        <ul>
            <li>transition matrix</li>
            <li>state matrix</li>
            <li>noise covariance matrix</li>
        </ul>
    </li>
</ul>
</p>