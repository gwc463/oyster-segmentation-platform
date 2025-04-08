package com.xai.oyster.service;

import com.xai.oyster.entity.Dataset;
import com.xai.oyster.repository.DatasetRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DatasetService {
    @Autowired
    private DatasetRepository datasetRepository;

    public Dataset createDataset(Dataset dataset) {
        return datasetRepository.save(dataset);
    }

    public List<Dataset> getAllDatasets() {
        return datasetRepository.findAll();
    }

    public void deleteDataset(Long id) {
        datasetRepository.deleteById(id);
    }
}