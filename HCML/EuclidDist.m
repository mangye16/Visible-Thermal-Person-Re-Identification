function D = EuclidDist(galX, probX)
    D = bsxfun(@plus, sum(galX.^2, 2), sum(probX.^2, 2)') - 2 * galX * probX';
end
