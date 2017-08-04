local XnorScaling, parent = torch.class('XnorScaling', 'nn.Module')

function XnorScaling:__init()
    parent.__init(self)
    self.norm = 1
end

function XnorScaling:updateOutput(input)
    self.norm = input:norm(1) / input:nElement() 
    self.output:resizeAs(input):copy(input):div(self.norm)
    return self.output
end

function XnorScaling:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput):copy(gradOutput):div(self.norm)
    return self.gradInput
end
