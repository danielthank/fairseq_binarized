local BinaryTemporalConvolution, parent =
    torch.class('BinaryTemporalConvolution', 'cudnn.TemporalConvolution')
local Convolution = cudnn.SpatialConvolution

function BinaryTemporalConvolution:__init(inputFrameSize, outputFrameSize, kH, dH, padH)
    parent.__init(self, inputFrameSize, outputFrameSize, kH, dH, padH)
    self.weightOrg = self.weight:clone()
end

function BinaryTemporalConvolution:updateOutput(input)
    self.weight:clamp(-1, 1)
    self.weightOrg:copy(self.weight)
    self.weight:sign()

    parent.updateOutput(self, input)
    return self.output
end

function BinaryTemporalConvolution:updateGradInput(input, gradOutput)
    parent.updateGradInput(self, input, gradOutput)
    self.weight:copy(self.weightOrg)
    return self.gradInput
end

