package com.xai.oyster.repository;

import com.xai.oyster.entity.Dataset;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DatasetRepository extends JpaRepository<Dataset, Long> {
}