function Wout = prodMat(W)

[~, ~ , Num] = size(W);

Wout = 1;

for i = 1 : Num
    Wout = W(:,:,i)*Wout;
end

end
