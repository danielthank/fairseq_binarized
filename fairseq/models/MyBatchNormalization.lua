local MyBatchNormalization, parent =
    torch.class('MyBatchNormalization', 'nn.BatchNormalization')

function MyBatchNormalization:updateOutput(input)
    local origSize = input:size()
    --print('[for] orig: ', origSize)
    input:view(input, -1, origSize[3])
    --print('[for] new: ', input:size())
    parent.updateOutput(self, input)
    self.output:view(self.output, origSize)
    --print('[for] recover: ', self.output:size())
    return self.output
end

function MyBatchNormalization:updateGradInput(input, gradOutput)
    origSize = gradOutput:size()
    gradOutput:view(gradOutput, -1, origSize[3])
    parent.updateGradInput(self, input, gradOutput)
    return self.gradInput
end
