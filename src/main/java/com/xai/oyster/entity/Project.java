package com.xai.oyster.entity;

import lombok.Data;
import javax.persistence.*;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "projects")
public class Project {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private String creator;
    private String description;
    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;
    private LocalDateTime createdAt;
}