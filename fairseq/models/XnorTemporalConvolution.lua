local XnorTemporalConvolution, parent =
    torch.class('XnorTemporalConvolution', 'cudnn.TemporalConvolution')


function XnorTemporalConvolution:__init(inputFrameSize, outputFrameSize, kH, dH, padH)
    parent.__init(self, inputFrameSize, outputFrameSize, kH, dH, padH)
    --Convolution.noBias(self)
    self.weightB = self.weight:clone()
    self.weightOrg = self.weight:clone()
end

function XnorTemporalConvolution:updateOutput(input)
    self.weight:clamp(-1, 1)
    self.weightOrg:copy(self.weight)
    self.weightB:copy(self.weightOrg):sign()
    self.weight:copy(self.weightB)
    parent.updateOutput(self, input)
    self.weight:copy(self.weightOrg)
    --print(self.output:size())
    return self.output
end

function XnorTemporalConvolution:updateGradInput(input, gradOutput)
    --print('mean: ', self.weight:mean())
    --print('std:  ', self.weight:std())
    self.weight:copy(self.weightB)
    parent.updateGradInput(self, input, gradOutput)
    self.weight:copy(self.weightOrg)
    return self.gradInput
end

