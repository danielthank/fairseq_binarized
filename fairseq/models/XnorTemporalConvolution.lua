local XnorTemporalConvolution, parent =
    torch.class('XnorTemporalConvolution', 'cudnn.TemporalConvolution')

function XnorTemporalConvolution:__init(inputFrameSize, outputFrameSize, kH, dH, padH)
    parent.__init(self, inputFrameSize, outputFrameSize, kH, dH, padH)
    self.weightOrg = self.weight:clone()
    self.norm = 0
end

function XnorTemporalConvolution:updateOutput(input)
    local mean = self.weight:mean()
    self.weight:csub(mean)
    self.weight:clamp(-1, 1)
    self.weightOrg:copy(self.weight)

    self.norm = self.weight:norm(1) / self.weight:nElement()
    self.weight:sign():mul(self.norm)

    parent.updateOutput(self, input)
    return self.output
end

function XnorTemporalConvolution:updateGradInput(input, gradOutput)
    parent.updateGradInput(self, input, gradOutput)
    local size = self.weight:size()
    local n = self.weight:nElement()
    local multi = self.norm + 1/n

    self.weight:copy(self.weightOrg)
    self.gradWeight:cmul(multi)
    return self.gradInput
end

