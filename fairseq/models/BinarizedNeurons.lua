local BinarizedNeurons,parent = torch.class('BinarizedNeurons', 'nn.Module')

function BinarizedNeurons:__init()
   parent.__init(self)
 end

function BinarizedNeurons:updateOutput(input)
    self.output:resizeAs(input):copy(input):sign()
    return self.output
end

function BinarizedNeurons:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    return self.gradInput
end
